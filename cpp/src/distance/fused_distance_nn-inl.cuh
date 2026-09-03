/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __FUSED_DISTANCE_NN_H
#define __FUSED_DISTANCE_NN_H

#pragma once

#include "detail/fused_distance_nn.cuh"
#include "fused_distance_nn_helpers.cuh"
#include "top_1_nn.cuh"
#include "unfused_distance_nn.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/linalg/map.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/util_type.cuh>

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace cuvs {
namespace distance {

/**
 * \ingroup fused_l2_nn
 * @{
 */

template <typename DataT, typename IdxT, typename NormT, typename ReduceOpT, typename KVPReduceOpT>
void fusedDistanceNN(IdxT* nearest_idx,
                     DataT* nearest_dist,
                     const DataT* x,
                     const DataT* y,
                     const NormT* xn,
                     const NormT* yn,
                     IdxT m,
                     IdxT n,
                     IdxT k,
                     void* workspace,
                     ReduceOpT redOp,
                     KVPReduceOpT pairRedOp,
                     bool sqrt,
                     bool initOutBuffer,
                     bool isRowMajor,
                     cuvs::distance::DistanceType metric,
                     float metric_arg,
                     raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_scratch,
                     cudaStream_t stream)
{
  ASSERT(isRowMajor, "fusedDistanceNN only supports row major inputs");
  bool is_skinny = k < 32;

  size_t bytes = sizeof(DataT) * k;
  auto px      = reinterpret_cast<uintptr_t>(x);
  auto py      = reinterpret_cast<uintptr_t>(y);
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0 && px % 16 == 0 && py % 16 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        NormT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        NormT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 16 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    }
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0 && px % 8 == 0 && py % 8 == 0) {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<
        DataT,
        NormT,
        IdxT,
        typename raft::linalg::Policy4x4Skinny<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    } else {
      detail::fusedDistanceNNImpl<
        DataT,
        NormT,
        IdxT,
        typename raft::linalg::Policy4x4<DataT, 8 / sizeof(DataT)>::Policy,
        ReduceOpT>(nearest_idx,
                   nearest_dist,
                   x,
                   y,
                   xn,
                   yn,
                   m,
                   n,
                   k,
                   (int*)workspace,
                   redOp,
                   pairRedOp,
                   sqrt,
                   initOutBuffer,
                   isRowMajor,
                   metric,
                   metric_arg,
                   cutlass_kvp_scratch,
                   stream);
    }
  } else {
    if (is_skinny) {
      detail::fusedDistanceNNImpl<DataT,
                                  NormT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4Skinny<DataT, 1>::Policy,
                                  ReduceOpT>(nearest_idx,
                                             nearest_dist,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             cutlass_kvp_scratch,
                                             stream);
    } else {
      detail::fusedDistanceNNImpl<DataT,
                                  NormT,
                                  IdxT,
                                  typename raft::linalg::Policy4x4<DataT, 1>::Policy,
                                  ReduceOpT>(nearest_idx,
                                             nearest_dist,
                                             x,
                                             y,
                                             xn,
                                             yn,
                                             m,
                                             n,
                                             k,
                                             (int*)workspace,
                                             redOp,
                                             pairRedOp,
                                             sqrt,
                                             initOutBuffer,
                                             isRowMajor,
                                             metric,
                                             metric_arg,
                                             cutlass_kvp_scratch,
                                             stream);
    }
  }
}

/**
 * @brief Fused GEMM + 1-NN minimum reduction.
 *
 * @param[out] nearest_idx   Nearest neighbor index per row, length `m` (optional).
 * @param[out] nearest_dist  Minimum distance per row, length `m` (optional, may be null).
 * @param[in]  cutlass_kvp_scratch Temp KVP buffer, length `m`, for index-bearing CUTLASS/SIMT
 *                                  output. Distance-only output may pass null and write directly
 *                                  to `nearest_dist`.
 */
template <typename DataT, typename IdxT, typename NormT = DataT>
void fusedDistanceNNMinReduce(IdxT* nearest_idx,
                              DataT* nearest_dist,
                              const DataT* x,
                              const DataT* y,
                              const NormT* xn,
                              const NormT* yn,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              void* workspace,
                              bool sqrt,
                              bool initOutBuffer,
                              bool isRowMajor,
                              cuvs::distance::DistanceType metric,
                              float metric_arg,
                              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_scratch,
                              cudaStream_t stream)
{
  MinAndDistanceReduceOp<IdxT, DataT> redOp;
  redOp.out_idx  = nearest_idx;
  redOp.out_dist = nearest_dist;
  KVPMinReduce<IdxT, DataT> pairRedOp;

  fusedDistanceNN<DataT, IdxT>(nearest_idx,
                               nearest_dist,
                               x,
                               y,
                               xn,
                               yn,
                               m,
                               n,
                               k,
                               workspace,
                               redOp,
                               pairRedOp,
                               sqrt,
                               initOutBuffer,
                               isRowMajor,
                               metric,
                               metric_arg,
                               cutlass_kvp_scratch,
                               stream);
}

template <typename DataT, typename IdxT, typename NormT>
void top_1_nn(raft::resources const& handle,
              IdxT* nearest_idx,
              DataT* nearest_dist,
              const DataT* x,
              const DataT* y,
              const NormT* xn,
              const NormT* yn,
              IdxT m,
              IdxT n,
              IdxT k,
              const detail::Top1nnTuning& tuning,
              void* workspace,
              std::size_t workspace_bytes,
              bool sqrt,
              bool init_out_buffer,
              bool is_row_major,
              cuvs::distance::DistanceType metric,
              float metric_arg,
              detail::Fused1nnBackend backend,
              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
              cudaStream_t stream)
{
  RAFT_EXPECTS(is_row_major, "fusedDistanceNN only supports row-major inputs");
  if (backend == detail::Fused1nnBackend::Cutile) {
    if constexpr (detail::is_fused_1nn_cutile_data_v<DataT> &&
                  std::is_same_v<NormT, detail::fused_1nn_cutile_norm_t<DataT>>) {
      const bool launched = detail::try_fused_1nn_tile<DataT, IdxT>(
        nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, sqrt, workspace, stream);
      RAFT_EXPECTS(launched,
                   "Requested cuTile fused 1-NN backend is unavailable for this input/device");
      return;
    }
    RAFT_FAIL("Requested cuTile fused 1-NN backend does not support these data/norm types");
  }
  RAFT_EXPECTS(detail::can_launch_fused_1nn_backend(backend, x, y, m, n, k, metric),
               "Requested fused 1-NN backend is unavailable for this input");
  RAFT_EXPECTS(metric != cuvs::distance::DistanceType::InnerProduct,
               "Only cuTile top_1_nn supports InnerProduct (as a maximum reduction)");
  constexpr bool matching_norm_type = std::is_same_v<NormT, DataT>;
  RAFT_EXPECTS(matching_norm_type, "CUTLASS and unfused top_1_nn require matching norm types");
  if constexpr (matching_norm_type) {
    if (backend == detail::Fused1nnBackend::Unfused) {
      RAFT_EXPECTS(cutlass_kvp_output != nullptr,
                   "Unfused top_1_nn requires its native KVP output buffer");
      RAFT_EXPECTS(tuning.unfused.row_tile > 0 && tuning.unfused.candidate_tile > 0,
                   "Unfused top_1_nn tile dimensions must be positive");

      const auto max_row_tile       = static_cast<std::size_t>(m);
      const auto max_candidate_tile = static_cast<std::size_t>(n);
      const auto row_tile = static_cast<IdxT>(std::min(tuning.unfused.row_tile, max_row_tile));
      const auto candidate_tile =
        static_cast<IdxT>(std::min(tuning.unfused.candidate_tile, max_candidate_tile));
      const auto required_workspace_bytes = static_cast<std::size_t>(row_tile) *
                                            static_cast<std::size_t>(candidate_tile) *
                                            sizeof(DataT);
      RAFT_EXPECTS(workspace != nullptr && workspace_bytes >= required_workspace_bytes,
                   "Unfused top_1_nn workspace is smaller than its configured tile");

      using KeyValueT = raft::KeyValuePair<IdxT, DataT>;
      rmm::device_uvector<KeyValueT> candidate_min(candidate_tile < n ? row_tile : 0, stream);
      for (IdxT row_offset = 0; row_offset < m; row_offset += row_tile) {
        const auto rows = std::min(row_tile, static_cast<IdxT>(m - row_offset));
        auto output =
          raft::make_device_vector_view<KeyValueT, IdxT>(cutlass_kvp_output + row_offset, rows);
        for (IdxT candidate_offset = 0; candidate_offset < n; candidate_offset += candidate_tile) {
          const auto candidates = std::min(candidate_tile, static_cast<IdxT>(n - candidate_offset));
          auto* tile_output = candidate_offset == 0 ? output.data_handle() : candidate_min.data();
          unfusedDistanceNNMinReduce<DataT, DataT, KeyValueT, IdxT>(
            handle,
            tile_output,
            x + row_offset * k,
            y + candidate_offset * k,
            xn + row_offset,
            yn + candidate_offset,
            rows,
            candidates,
            k,
            workspace,
            sqrt,
            candidate_offset != 0 || init_out_buffer,
            is_row_major,
            metric,
            metric_arg,
            stream);
          if (candidate_offset != 0) {
            auto candidate_output =
              raft::make_device_vector_view<const KeyValueT, IdxT>(candidate_min.data(), rows);
            raft::linalg::map(
              handle,
              output,
              [candidate_offset] __device__(KeyValueT current, KeyValueT candidate) {
                candidate.key += candidate_offset;
                return candidate.value < current.value ? candidate : current;
              },
              raft::make_const_mdspan(output),
              candidate_output);
          }
        }
      }
      return;
    }
    RAFT_EXPECTS(backend == detail::Fused1nnBackend::Cutlass, "Unknown fused 1-NN backend");
    RAFT_EXPECTS(cutlass_kvp_output != nullptr,
                 "CUTLASS fused 1-NN requires its native KVP output buffer");
    MinAndDistanceReduceOp<IdxT, DataT> red_op;
    KVPMinReduce<IdxT, DataT> pair_red_op;
    fusedDistanceNN<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(cutlass_kvp_output,
                                                                  x,
                                                                  y,
                                                                  xn,
                                                                  yn,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  workspace,
                                                                  red_op,
                                                                  pair_red_op,
                                                                  sqrt,
                                                                  init_out_buffer,
                                                                  is_row_major,
                                                                  metric,
                                                                  metric_arg,
                                                                  stream);
  }
}

template <typename DataT, typename IdxT, typename NormT>
void top_1_nn(IdxT* nearest_idx,
              DataT* nearest_dist,
              const DataT* x,
              const DataT* y,
              const NormT* xn,
              const NormT* yn,
              IdxT m,
              IdxT n,
              IdxT k,
              const detail::Top1nnTuning&,
              void* workspace,
              std::size_t,
              bool sqrt,
              bool init_out_buffer,
              bool is_row_major,
              cuvs::distance::DistanceType metric,
              float metric_arg,
              detail::Fused1nnBackend backend,
              raft::KeyValuePair<IdxT, DataT>* cutlass_kvp_output,
              cudaStream_t stream)
{
  RAFT_EXPECTS(backend == detail::Fused1nnBackend::Cutlass,
               "The no-handle top_1_nn compatibility path supports CUTLASS only");
  RAFT_EXPECTS(is_row_major, "fusedDistanceNN only supports row-major inputs");
  RAFT_EXPECTS(detail::can_launch_fused_1nn_backend(backend, x, y, m, n, k, metric),
               "Requested CUTLASS fused 1-NN backend is unavailable for this input");
  constexpr bool matching_norm_type = std::is_same_v<NormT, DataT>;
  RAFT_EXPECTS(matching_norm_type, "CUTLASS top_1_nn requires matching norm types");

  if constexpr (std::is_same_v<NormT, DataT>) {
    MinAndDistanceReduceOp<IdxT, DataT> red_op;
    KVPMinReduce<IdxT, DataT> pair_red_op;
    if (cutlass_kvp_output != nullptr) {
      fusedDistanceNN<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(cutlass_kvp_output,
                                                                    x,
                                                                    y,
                                                                    xn,
                                                                    yn,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    workspace,
                                                                    red_op,
                                                                    pair_red_op,
                                                                    sqrt,
                                                                    init_out_buffer,
                                                                    is_row_major,
                                                                    metric,
                                                                    metric_arg,
                                                                    stream);
      return;
    }
    RAFT_EXPECTS(nearest_idx == nullptr && nearest_dist != nullptr,
                 "CUTLASS scalar top_1_nn requires a distance output and no index output");
    fusedDistanceNN<DataT, DataT, IdxT>(nearest_dist,
                                        x,
                                        y,
                                        xn,
                                        yn,
                                        m,
                                        n,
                                        k,
                                        workspace,
                                        red_op,
                                        pair_red_op,
                                        sqrt,
                                        init_out_buffer,
                                        is_row_major,
                                        metric,
                                        metric_arg,
                                        stream);
  }
}

/** @} */

}  // namespace distance
}  // namespace cuvs

#endif
