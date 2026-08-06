/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Portable warp-level TF32 MMA backend.

__device__ __forceinline__ void load_tf32_candidate(float* dst,
                                                    const float* src,
                                                    int valid_dims,
                                                    int lane_id)
{
  for (int idx = lane_id; idx < Tf32MmaPolicy::K_TILE; idx += raft::warp_size()) {
    if (idx < valid_dims) {
      dst[idx] = nvcuda::wmma::__float_to_tf32(src[idx]);
    } else {
      dst[idx] = 0.0f;
    }
  }
}

template <int CtaThreads, typename Index_t>
__device__ __forceinline__ void compute_tf32_wmma_tile(float* candidates,
                                                       void* output,
                                                       const float* data,
                                                       int data_dim,
                                                       const Index_t* new_neighbors,
                                                       int list_new_size,
                                                       const Index_t* old_neighbors,
                                                       int list_old_size)
{
  using namespace nvcuda;
  using Policy        = Tf32MmaPolicy;
  using operand_t     = typename Policy::fragment_operand_t;
  using accumulator_t = typename Policy::accumulator_t;

  constexpr int NUM_MMA_WARPS      = CtaThreads / raft::warp_size();
  constexpr int OUTPUT_ROW_TILES   = MAX_NUM_BI_SAMPLES / WMMA_M;
  constexpr int OUTPUT_COL_TILES   = FUSED_DISTANCE_COLS / WMMA_N;
  constexpr int WARPS_PER_ROW      = NUM_MMA_WARPS / OUTPUT_ROW_TILES;
  constexpr int FRAGMENTS_PER_WARP = OUTPUT_COL_TILES / WARPS_PER_ROW;
  static_assert(CtaThreads == SM90_TF32_BLOCK_SIZE || CtaThreads == SM100_TF32_BLOCK_SIZE ||
                CtaThreads == PORTABLE_TF32_BLOCK_SIZE);
  static_assert(NUM_MMA_WARPS == OUTPUT_ROW_TILES * WARPS_PER_ROW);
  static_assert(OUTPUT_COL_TILES % WARPS_PER_ROW == 0);

  int warp_id          = threadIdx.x / raft::warp_size();
  int lane_id          = threadIdx.x % raft::warp_size();
  int row_tile         = warp_id / WARPS_PER_ROW;
  int active_fragments = list_old_size > 0 ? FRAGMENTS_PER_WARP : FRAGMENTS_PER_WARP / 2;
  int first_col_tile   = (warp_id % WARPS_PER_ROW) * active_fragments;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, Policy::K, operand_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, Policy::K, operand_t, wmma::col_major>
    b_frag[FRAGMENTS_PER_WARP];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, Policy::K, accumulator_t>
    accum[FRAGMENTS_PER_WARP];

#pragma unroll
  for (int fragment = 0; fragment < active_fragments; ++fragment) {
    wmma::fill_fragment(accum[fragment], accumulator_t{0});
  }

  // Inactive rows remain inactive for the complete dot product. Initialize them once instead of
  // rewriting their zero padding for every K tile.
  for (int candidate = warp_id; candidate < FUSED_DISTANCE_COLS; candidate += NUM_MMA_WARPS) {
    bool is_new       = candidate < MAX_NUM_BI_SAMPLES;
    int candidate_idx = is_new ? candidate : candidate - MAX_NUM_BI_SAMPLES;
    bool is_active    = is_new ? candidate_idx < list_new_size : candidate_idx < list_old_size;
    if (!is_active) {
      load_tf32_candidate(
        candidates + candidate * Policy::K_LD, static_cast<const float*>(nullptr), 0, lane_id);
    }
  }
  __syncthreads();

  for (int data_base = 0; data_base < data_dim; data_base += Policy::K_TILE) {
    int valid_dims = min(Policy::K_TILE, data_dim - data_base);
    for (int candidate = warp_id; candidate < FUSED_DISTANCE_COLS; candidate += NUM_MMA_WARPS) {
      bool is_new       = candidate < MAX_NUM_BI_SAMPLES;
      int candidate_idx = is_new ? candidate : candidate - MAX_NUM_BI_SAMPLES;
      bool is_active    = is_new ? candidate_idx < list_new_size : candidate_idx < list_old_size;
      if (is_active) {
        Index_t neighbor_id = is_new ? new_neighbors[candidate_idx] : old_neighbors[candidate_idx];
        const float* src    = data + static_cast<size_t>(neighbor_id) * data_dim + data_base;
        load_tf32_candidate(candidates + candidate * Policy::K_LD, src, valid_dims, lane_id);
      }
    }
    __syncthreads();

    for (int k_base = 0; k_base < Policy::K_TILE; k_base += Policy::K) {
      wmma::load_matrix_sync(
        a_frag, candidates + row_tile * WMMA_M * Policy::K_LD + k_base, Policy::K_LD);
#pragma unroll
      for (int fragment = 0; fragment < active_fragments; ++fragment) {
        int col_tile = first_col_tile + fragment;
        wmma::load_matrix_sync(
          b_frag[fragment], candidates + col_tile * WMMA_N * Policy::K_LD + k_base, Policy::K_LD);
        wmma::mma_sync(accum[fragment], a_frag, b_frag[fragment], accum[fragment]);
      }
    }
    __syncthreads();
  }

  auto* typed_output = reinterpret_cast<accumulator_t*>(output);
#pragma unroll
  for (int fragment = 0; fragment < active_fragments; ++fragment) {
    int col_tile = first_col_tile + fragment;
    wmma::store_matrix_sync(
      typed_output + row_tile * WMMA_M * FUSED_DISTANCE_LD + col_tile * WMMA_N,
      accum[fragment],
      FUSED_DISTANCE_LD,
      wmma::mem_row_major);
  }
}
