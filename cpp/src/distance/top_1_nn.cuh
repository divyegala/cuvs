/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/fused_distance_nn.cuh"

#include <cuvs/core/export.hpp>

#include <raft/core/kvp.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

namespace cuvs::distance {

/** Separate index and distance arrays used by backends with structure-of-arrays output. */
template <typename IdxT, typename DistT>
struct Top1nnOutput {
  IdxT* nearest_idx;
  DistT* nearest_dist;
};

namespace detail {

template <typename DataT, typename IdxT>
struct Top1nnOutputTypes {
  using kvp      = raft::KeyValuePair<IdxT, DataT>*;
  using scalar   = DataT*;
  using separate = Top1nnOutput<IdxT, DataT>;
};

}  // namespace detail

/** Dispatch 1-NN to a selected backend using its native output representation. */
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
                          detail::Top1nnBackend backend,
                          cudaStream_t stream);

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
    detail::Top1nnBackend,                                                                   \
    cudaStream_t)

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
