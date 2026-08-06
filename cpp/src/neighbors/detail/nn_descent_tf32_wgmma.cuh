/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SM90 WGMMA backend for the fused TF32 kernel.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
CUTE_DEVICE auto make_sm90_tf32_tiled_mma()
{
  return cute::make_tiled_mma(cute::SM90_64x128x8_F32TF32TF32_SS_TN{});
}

template <typename SmemTensor>
CUTE_DEVICE void store_wgmma_u64(SmemTensor& smem, int row, int k, uint2 const& bits)
{
  auto* dst = &smem(row, k);
  cutlass::arch::shared_store<8>(cute::cast_smem_ptr_to_uint(dst), &bits);
}

CUTE_DEVICE uint2 pack_wgmma_tf32x2(float2 values)
{
  cute::tfloat32_t x{values.x};
  cute::tfloat32_t y{values.y};
  return uint2{x.raw(), y.raw()};
}

template <typename Index_t>
CUTE_DEVICE void compute_tf32_wgmma_tile(void* workspace,
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

  constexpr auto smem_shape =
    cute::make_shape(cute::Int<FUSED_DISTANCE_COLS>{}, cute::Int<Policy::K_TILE>{});
  using SmemLayout =
    decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<element_t>{}, smem_shape));

  auto s_candidates =
    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<element_t*>(workspace)), SmemLayout{});
  auto s_a =
    cute::local_tile(s_candidates,
                     cute::make_shape(cute::Int<MAX_NUM_BI_SAMPLES>{}, cute::Int<Policy::K_TILE>{}),
                     cute::make_coord(cute::_0{}, cute::_0{}));
  auto s_c = cute::make_tensor(
    cute::make_smem_ptr(reinterpret_cast<accumulator_t*>(workspace)),
    cute::make_layout(
      cute::make_shape(cute::Int<MAX_NUM_BI_SAMPLES>{}, cute::Int<FUSED_DISTANCE_COLS>{}),
      cute::make_stride(cute::Int<FUSED_DISTANCE_LD>{}, cute::_1{})));

  auto mma        = make_sm90_tf32_tiled_mma();
  auto thread_mma = mma.get_slice(threadIdx.x);
  auto tCsA       = thread_mma.partition_A(s_a);
  auto tCsB       = thread_mma.partition_B(s_candidates);
  auto tCsC       = thread_mma.partition_C(s_c);
  auto tCrA       = thread_mma.make_fragment_A(tCsA);
  auto tCrB       = thread_mma.make_fragment_B(tCsB);
  auto tCrC       = thread_mma.make_fragment_C(tCsC);
  cute::clear(tCrC);

  int warp_id = threadIdx.x / raft::warp_size();
  int lane_id = threadIdx.x % raft::warp_size();

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
      store_wgmma_u64(s_candidates, candidate, k0, pack_wgmma_tf32x2(values));
    }

    cutlass::arch::fence_view_async_shared();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    cute::warpgroup_fence_operand(tCrC);
    cute::warpgroup_arrive();
    cute::gemm(mma, tCrA, tCrB, tCrC);
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(tCrC);
    asm volatile("bar.sync 1, 128;" ::: "memory");
  }

  cute::copy(tCrC, tCsC);
}
#endif
