/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)

template <typename Policy>
CUTE_DEVICE auto make_sm100_tiled_mma()
{
  using element_t     = std::conditional_t<Policy::IS_TF32, cute::tfloat32_t, cute::half_t>;
  using accumulator_t = typename Policy::accumulator_t;

  if constexpr (Policy::IS_TF32) {
    return cute::make_tiled_mma(cute::SM100_MMA_TF32_SS<element_t,
                                                        element_t,
                                                        accumulator_t,
                                                        64,
                                                        128,
                                                        cute::UMMA::Major::K,
                                                        cute::UMMA::Major::K>{});
  } else {
    return cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<element_t,
                                                           element_t,
                                                           accumulator_t,
                                                           64,
                                                           128,
                                                           cute::UMMA::Major::K,
                                                           cute::UMMA::Major::K>{});
  }
}

template <typename Policy>
struct Sm100MmaWorkspaceSize {
  using element_t            = std::conditional_t<Policy::IS_TF32, cute::tfloat32_t, cute::half_t>;
  static constexpr int value = (FUSED_DISTANCE_COLS * 2 + MAX_NUM_BI_SAMPLES) * Policy::K_TILE *
                               static_cast<int>(sizeof(element_t));
};

template <typename Policy, typename Data_t, typename Index_t>
CUTE_DEVICE void compute_fused_tcgen05_tile(void* workspace,
                                            cute::uint64_t& mma_barrier,
                                            cute::uint32_t& tmem_base_ptr,
                                            const Data_t* data,
                                            int data_dim,
                                            const Index_t* new_neighbors,
                                            int list_new_size,
                                            const Index_t* old_neighbors,
                                            int list_old_size)
{
  using element_t     = std::conditional_t<Policy::IS_TF32, cute::tfloat32_t, cute::half_t>;
  using accumulator_t = typename Policy::accumulator_t;

  auto mma = make_sm100_tiled_mma<Policy>();
  constexpr auto mma_tile_shape =
    cute::make_shape(cute::Int<64>{}, cute::Int<128>{}, cute::Int<Policy::K_TILE>{});
  constexpr auto mma_shape_a = cute::partition_shape_A(mma, cute::select<0, 2>(mma_tile_shape));
  constexpr auto mma_shape_b = cute::partition_shape_B(mma, cute::select<1, 2>(mma_tile_shape));
  using SmemLayoutA          = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<element_t>{}, mma_shape_a));
  using SmemLayoutB          = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<element_t>{}, mma_shape_b));

  constexpr int RAW_BYTES =
    FUSED_DISTANCE_COLS * Policy::K_TILE * static_cast<int>(sizeof(element_t));
  constexpr int A_BYTES = cute::cosize_v<SmemLayoutA> * static_cast<int>(sizeof(element_t));
  auto* workspace_bytes = reinterpret_cast<unsigned char*>(workspace);

  auto s_raw_b = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace_bytes)),
    cute::make_layout(cute::make_shape(cute::Int<128>{}, cute::Int<Policy::K_TILE>{}),
                      cute::make_stride(cute::Int<Policy::K_TILE>{}, cute::_1{})));
  auto s_raw_a = cute::local_tile(s_raw_b,
                                  cute::make_shape(cute::Int<64>{}, cute::Int<Policy::K_TILE>{}),
                                  cute::make_coord(cute::_0{}, cute::_0{}));
  auto tCsA    = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace_bytes + RAW_BYTES)), SmemLayoutA{});
  auto tCsB = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace_bytes + RAW_BYTES + A_BYTES)),
    SmemLayoutB{});
  auto s_c = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<accumulator_t*>(workspace)),
    cute::make_layout(cute::make_shape(cute::Int<64>{}, cute::Int<128>{}),
                      cute::make_stride(cute::Int<FUSED_DISTANCE_LD>{}, cute::_1{})));

  auto cta_mma = mma.get_slice(cute::_0{});
  auto tCgA    = cta_mma.partition_A(s_raw_a);
  auto tCgB    = cta_mma.partition_B(s_raw_b);
  auto tCgC    = cta_mma.partition_C(s_c);
  auto tCrA    = cta_mma.make_fragment_A(tCsA);
  auto tCrB    = cta_mma.make_fragment_B(tCsB);
  auto tCtAcc  = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = threadIdx.x / raft::warp_size() == 0;
  using TmemAllocator     = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = tmem_base_ptr;

  if (elect_one_warp && elect_one_thr) { cute::initialize_barrier(mma_barrier, 1); }
  __syncthreads();

  mma.accumulate_       = cute::UMMA::ScaleOut::Zero;
  int mma_barrier_phase = 0;
  for (int data_base = 0; data_base < data_dim; data_base += Policy::K_TILE) {
    int valid_dims = min(Policy::K_TILE, data_dim - data_base);
    for (int linear = threadIdx.x; linear < FUSED_DISTANCE_COLS * Policy::K_TILE;
         linear += blockDim.x) {
      int candidate     = linear / Policy::K_TILE;
      int k             = linear % Policy::K_TILE;
      bool is_new       = candidate < MAX_NUM_BI_SAMPLES;
      int candidate_idx = is_new ? candidate : candidate - MAX_NUM_BI_SAMPLES;
      bool is_active    = is_new ? candidate_idx < list_new_size : candidate_idx < list_old_size;

      element_t value{};
      if (is_active && k < valid_dims) {
        Index_t neighbor_id = is_new ? new_neighbors[candidate_idx] : old_neighbors[candidate_idx];
        value =
          static_cast<element_t>(data[static_cast<size_t>(neighbor_id) * data_dim + data_base + k]);
      }
      s_raw_b(candidate, k) = value;
    }
    __syncthreads();

    for (int linear = threadIdx.x; linear < cute::size(tCsA); linear += blockDim.x) {
      tCsA[linear] = tCgA[linear];
    }
    for (int linear = threadIdx.x; linear < cute::size(tCsB); linear += blockDim.x) {
      tCsB[linear] = tCgB[linear];
    }
    __syncthreads();

    if (elect_one_warp) {
      for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
        cute::gemm(mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCtAcc);
        mma.accumulate_ = cute::UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&mma_barrier);
    }
    cute::wait_barrier(mma_barrier, mma_barrier_phase);
    mma_barrier_phase ^= 1;
  }

  if (threadIdx.x < 64) {
    auto tmem_copy   = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_16dp32b1x{}, tCtAcc);
    auto thread_copy = tmem_copy.get_slice(threadIdx.x);
    auto tDtAcc      = thread_copy.partition_S(tCtAcc);
    auto tDsC        = thread_copy.partition_D(tCgC);
    auto tDrAcc      = cute::make_tensor<accumulator_t>(cute::shape(tDsC));
    cute::copy(tmem_copy, tDtAcc, tDrAcc);
    cute::copy(tDrAcc, tDsC);
  }
  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

#endif
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
