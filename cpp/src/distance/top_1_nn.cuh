/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/fused_distance_nn.cuh"

#include <cuvs/core/export.hpp>

#include <raft/core/kvp.hpp>
#include <raft/core/resources.hpp>

#include <type_traits>

namespace cuvs::distance {

/** Separate index and distance arrays used by backends with structure-of-arrays output. */
template <typename IdxT, typename DistT>
struct Top1nnOutput {
  IdxT* nearest_idx;
  DistT* nearest_dist;
};

namespace detail {

template <typename DataT>
using top_1_nn_distance_t = std::conditional_t<std::is_same_v<DataT, half>, float, DataT>;

template <typename DataT, typename IdxT>
struct Top1nnOutputTypes {
  using kvp      = raft::KeyValuePair<IdxT, DataT>*;
  using scalar   = DataT*;
  using separate = Top1nnOutput<IdxT, top_1_nn_distance_t<DataT>>;
};

}  // namespace detail

/**
 * Return the workspace bytes required for one top-1 NN call.
 *
 * Callers that batch a larger problem should pass their maximum batch dimensions and reuse one
 * allocation across calls.
 */
template <typename DataT, typename IdxT>
CUVS_EXPORT std::size_t top_1_nn_workspace_size(IdxT m,
                                                IdxT n,
                                                const detail::Top1nnTuning& tuning,
                                                detail::Top1nnBackend backend);

/** Resolve one exact invocation and return its native storage requirements. */
template <typename DataT, typename IdxT>
CUVS_EXPORT detail::Top1nnPlan<IdxT> probe_top_1_nn(
  raft::resources const& handle,
  const DataT* x,
  const DataT* y,
  IdxT m,
  IdxT n,
  IdxT k,
  const detail::Top1nnTuning& tuning,
  DistanceType metric,
  detail::Top1nnBackend backend = detail::Top1nnBackend::Auto);

/** Dispatch 1-NN using the native output representation recorded in a reusable plan. */
template <typename DataT, typename IdxT, typename OutputT, typename NormT = DataT>
CUVS_EXPORT void top_1_nn(raft::resources const& handle,
                          OutputT output,
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
                          DistanceType metric,
                          float metric_arg,
                          const detail::Top1nnPlan<IdxT>& plan);

/** Convenience overload for an explicit backend or AUTO request. */
template <typename DataT, typename IdxT, typename OutputT, typename NormT = DataT>
void top_1_nn(raft::resources const& handle,
              OutputT output,
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
              DistanceType metric,
              float metric_arg,
              detail::Top1nnBackend backend)
{
  const auto plan = probe_top_1_nn(handle, x, y, m, n, k, tuning, metric, backend);
  RAFT_EXPECTS(plan.available, "No top_1_nn backend is available for this invocation");
  top_1_nn(handle,
           output,
           x,
           y,
           xn,
           yn,
           m,
           n,
           k,
           tuning,
           workspace,
           workspace_bytes,
           sqrt,
           init_out_buffer,
           is_row_major,
           metric,
           metric_arg,
           plan);
}
#define CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(DataT, IdxT)            \
  extern template std::size_t top_1_nn_workspace_size<DataT, IdxT>( \
    IdxT, IdxT, const detail::Top1nnTuning&, detail::Top1nnBackend)

CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(float, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(float, int64_t);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(double, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(double, int64_t);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(half, int);
CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE(half, int64_t);

#undef CUVS_EXTERN_TOP_1_NN_WORKSPACE_SIZE

#define CUVS_EXTERN_PROBE_TOP_1_NN(DataT, IdxT)                         \
  extern template detail::Top1nnPlan<IdxT> probe_top_1_nn<DataT, IdxT>( \
    raft::resources const&,                                             \
    const DataT*,                                                       \
    const DataT*,                                                       \
    IdxT,                                                               \
    IdxT,                                                               \
    IdxT,                                                               \
    const detail::Top1nnTuning&,                                        \
    DistanceType,                                                       \
    detail::Top1nnBackend)

CUVS_EXTERN_PROBE_TOP_1_NN(float, int);
CUVS_EXTERN_PROBE_TOP_1_NN(float, int64_t);
CUVS_EXTERN_PROBE_TOP_1_NN(double, int);
CUVS_EXTERN_PROBE_TOP_1_NN(double, int64_t);
CUVS_EXTERN_PROBE_TOP_1_NN(half, int);
CUVS_EXTERN_PROBE_TOP_1_NN(half, int64_t);

#undef CUVS_EXTERN_PROBE_TOP_1_NN

#define CUVS_EXTERN_TOP_1_NN(DataT, IdxT, NormT, OutputKind)                                 \
  extern template void                                                                       \
  top_1_nn<DataT, IdxT, typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind, NormT>( \
    raft::resources const&,                                                                  \
    typename detail::Top1nnOutputTypes<DataT, IdxT>::OutputKind,                             \
    const DataT*,                                                                            \
    const DataT*,                                                                            \
    const NormT*,                                                                            \
    const NormT*,                                                                            \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    IdxT,                                                                                    \
    const detail::Top1nnTuning&,                                                             \
    void*,                                                                                   \
    std::size_t,                                                                             \
    bool,                                                                                    \
    bool,                                                                                    \
    bool,                                                                                    \
    DistanceType,                                                                            \
    float,                                                                                   \
    const detail::Top1nnPlan<IdxT>&)

CUVS_EXTERN_TOP_1_NN(float, int, float, kvp);
CUVS_EXTERN_TOP_1_NN(float, int, float, scalar);
CUVS_EXTERN_TOP_1_NN(float, int, float, separate);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, kvp);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, scalar);
CUVS_EXTERN_TOP_1_NN(float, int64_t, float, separate);
CUVS_EXTERN_TOP_1_NN(double, int, double, kvp);
CUVS_EXTERN_TOP_1_NN(double, int, double, scalar);
CUVS_EXTERN_TOP_1_NN(double, int64_t, double, kvp);
CUVS_EXTERN_TOP_1_NN(double, int64_t, double, scalar);
CUVS_EXTERN_TOP_1_NN(half, int, float, separate);
CUVS_EXTERN_TOP_1_NN(half, int64_t, float, separate);

#undef CUVS_EXTERN_TOP_1_NN

}  // namespace cuvs::distance
