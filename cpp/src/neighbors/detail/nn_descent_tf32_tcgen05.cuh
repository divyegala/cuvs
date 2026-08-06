/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SM100 TCGen05 backend for the fused TF32 kernel.

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)

CUTE_DEVICE auto make_sm100_tf32_tiled_mma()
{
  return cute::make_tiled_mma(cute::SM100_MMA_TF32_SS<cute::tfloat32_t,
                                                      cute::tfloat32_t,
                                                      float,
                                                      64,
                                                      128,
                                                      cute::UMMA::Major::K,
                                                      cute::UMMA::Major::K>{});
}

struct Sm100Tf32WorkspaceSize {
  using element_t            = cute::tfloat32_t;
  static constexpr int value = (FUSED_DISTANCE_COLS + MAX_NUM_BI_SAMPLES) * Tf32MmaPolicy::K_TILE *
                               static_cast<int>(sizeof(element_t));
};

template <int Rows, typename SmemTensor>
CUTE_DEVICE void store_tcgen05_u64(SmemTensor& smem, int row, int k, uint2 const& bits)
{
  // Scalar coordinates unflatten the hierarchical UMMA layout in logical row-first order.
  auto* dst = &smem[row + Rows * k];
  cutlass::arch::shared_store<8>(cute::cast_smem_ptr_to_uint(dst), &bits);
}

CUTE_DEVICE uint2 pack_tcgen05_tf32x2(float2 values)
{
  cute::tfloat32_t x{values.x};
  cute::tfloat32_t y{values.y};
  return uint2{x.raw(), y.raw()};
}

template <typename Index_t>
CUTE_DEVICE void compute_tf32_tcgen05_tile(void* workspace,
                                           cute::uint64_t& mma_barrier,
                                           cute::uint32_t& tmem_base_ptr,
                                           const float* data,
                                           int data_dim,
                                           const Index_t* new_neighbors,
                                           int list_new_size,
                                           const Index_t* old_neighbors,
                                           int list_old_size)
{
  using Policy        = Tf32MmaPolicy;
  using element_t     = cute::tfloat32_t;
  using accumulator_t = float;

  auto mma = make_sm100_tf32_tiled_mma();
  constexpr auto mma_tile_shape =
    cute::make_shape(cute::Int<64>{}, cute::Int<128>{}, cute::Int<Policy::K_TILE>{});
  constexpr auto mma_shape_a = cute::partition_shape_A(mma, cute::select<0, 2>(mma_tile_shape));
  constexpr auto mma_shape_b = cute::partition_shape_B(mma, cute::select<1, 2>(mma_tile_shape));
  using SmemLayoutA          = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<element_t>{}, mma_shape_a));
  using SmemLayoutB          = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<element_t>{}, mma_shape_b));

  constexpr int A_BYTES = cute::cosize_v<SmemLayoutA> * static_cast<int>(sizeof(element_t));
  auto* workspace_bytes = reinterpret_cast<unsigned char*>(workspace);

  auto tCsA = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace_bytes)),
                                SmemLayoutA{});
  auto tCsB = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace_bytes + A_BYTES)), SmemLayoutB{});
  auto s_c = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<accumulator_t*>(workspace)),
    cute::make_layout(cute::make_shape(cute::Int<64>{}, cute::Int<128>{}),
                      cute::make_stride(cute::Int<FUSED_DISTANCE_LD>{}, cute::_1{})));

  auto cta_mma = mma.get_slice(cute::_0{});
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
  int warp_id           = threadIdx.x / raft::warp_size();
  int lane_id           = threadIdx.x % raft::warp_size();

  for (int data_base = 0; data_base < data_dim; data_base += Policy::K_TILE) {
    int valid_dims = min(Policy::K_TILE, data_dim - data_base);
    for (int candidate = warp_id; candidate < FUSED_DISTANCE_COLS; candidate += 4) {
      bool is_new       = candidate < MAX_NUM_BI_SAMPLES;
      int candidate_idx = is_new ? candidate : candidate - MAX_NUM_BI_SAMPLES;
      bool is_active    = is_new ? candidate_idx < list_new_size : candidate_idx < list_old_size;
      const float* src  = nullptr;

      if (is_active) {
        Index_t neighbor_id = is_new ? new_neighbors[candidate_idx] : old_neighbors[candidate_idx];
        src                 = data + static_cast<size_t>(neighbor_id) * data_dim + data_base;
      }

      int k0        = lane_id * 2;
      float2 values = {};
      if (is_active) {
        if (k0 + 2 <= valid_dims && (reinterpret_cast<size_t>(src + k0) % alignof(float2)) == 0) {
          values = *reinterpret_cast<const float2*>(src + k0);
        } else {
          if (k0 < valid_dims) { values.x = src[k0]; }
          if (k0 + 1 < valid_dims) { values.y = src[k0 + 1]; }
        }
      }
      uint2 packed = pack_tcgen05_tf32x2(values);

      store_tcgen05_u64<FUSED_DISTANCE_COLS>(tCsB, candidate, k0, packed);
      if (is_new) { store_tcgen05_u64<MAX_NUM_BI_SAMPLES>(tCsA, candidate, k0, packed); }
    }

    cutlass::arch::fence_view_async_shared();
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
