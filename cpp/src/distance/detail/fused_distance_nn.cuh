/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "distance_ops/l2_exp.cuh"  // ops::l2_exp_distance_op
#include "fused_distance_nn/cutile/fused_1nn_tile.hpp"
#include "fused_distance_nn/cutlass_base.cuh"
#include "fused_distance_nn/fused_cosine_nn.cuh"
#include "fused_distance_nn/fused_l2_nn.cuh"
#include "fused_distance_nn/helper_structs.cuh"
#include "fused_distance_nn/simt_kernel.cuh"
#include "pairwise_distance_base.cuh"  // PairwiseDistances
#include <cuvs/distance/distance.hpp>
#include <raft/core/error.hpp>
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits
#include <type_traits>

namespace cuvs {
namespace distance {

namespace detail {

/** Backend selected by the fused 1-NN primitive. */
enum class Fused1nnBackend : std::uint8_t {
  Unfused = 0,
  FusedCutile,
  FusedCutlass,
};

template <typename IdxT>
constexpr Fused1nnBackend fused_1nn_legacy_backend(
  int cc_major, IdxT m, IdxT n, cuvs::distance::DistanceType metric)
{
  if (metric == cuvs::distance::DistanceType::InnerProduct) { return Fused1nnBackend::Unfused; }
  if (cc_major <= 8 || (cc_major == 9 && (m >= 4096 || n >= 4096))) {
    return Fused1nnBackend::FusedCutlass;
  }
  return Fused1nnBackend::Unfused;
}

/** Resolve the executable fused 1-NN backend before callers allocate result storage. */
template <typename DataT, typename IdxT>
Fused1nnBackend resolve_fused_1nn_backend(const raft::resources& handle,
                                          const DataT* x,
                                          const DataT* y,
                                          IdxT m,
                                          IdxT n,
                                          IdxT k,
                                          cuvs::distance::DistanceType metric)
{
  if constexpr (is_fused_1nn_cutile_data_v<DataT>) {
    if (can_launch_fused_1nn_tile(x, y, m, n, k, metric)) {
      return Fused1nnBackend::FusedCutile;
    }
  }
  const auto prop = raft::resource::get_device_properties(handle);
  return fused_1nn_legacy_backend(prop.major, m, n, metric);
}

template <typename DataT,
          typename NormT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedDistanceNNImpl(IdxT* nearest_idx,
                         DataT* nearest_dist,
                         const DataT* x,
                         const DataT* y,
                         const NormT* xn,
                         const NormT* yn,
                         IdxT m,
                         IdxT n,
                         IdxT k,
                         int* workspace,
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
  typedef Policy P;
  typedef raft::KeyValuePair<IdxT, DataT> KVP;
  constexpr auto maxVal = std::numeric_limits<DataT>::max();

  if constexpr (is_fused_1nn_cutile_data_v<DataT> &&
                std::is_same_v<fused_1nn_cutile_norm_t<DataT>, NormT>) {
    if constexpr (cuvs::detail::jit_lto::library_built_with_cutile()) {
      if (try_fused_1nn_tile<DataT, IdxT>(
            nearest_idx, nearest_dist, x, y, xn, yn, m, n, k, metric, sqrt, workspace, stream)) {
        return;
      }
    }
  }

  // InnerProduct is a cuTile-only specialization of this fused primitive. Callers that cannot
  // launch cuTile must select their unfused InnerProduct path instead of returning an untouched
  // sentinel from the legacy L2/cosine implementation below.
  RAFT_EXPECTS(metric != cuvs::distance::DistanceType::InnerProduct,
               "Fused InnerProduct 1-NN requires a compatible cuTile launcher");
  RAFT_EXPECTS((std::is_same_v<NormT, DataT>),
               "Mixed-precision norm inputs require a compatible cuTile launcher");

  const auto* legacy_xn = reinterpret_cast<const DataT*>(xn);
  const auto* legacy_yn = reinterpret_cast<const DataT*>(yn);

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));

  auto launch_legacy = [&]<typename OutT>(OutT* out, auto cutlass_red_op) {
    switch (metric) {
      case cuvs::distance::DistanceType::CosineExpanded:
        fusedCosineNN<DataT, OutT, IdxT, P, decltype(cutlass_red_op), KVPReduceOpT>(nearest_idx,
                                                                                    nearest_dist,
                                                                                    x,
                                                                                    y,
                                                                                    legacy_xn,
                                                                                    legacy_yn,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    workspace,
                                                                                    cutlass_red_op,
                                                                                    pairRedOp,
                                                                                    sqrt,
                                                                                    out,
                                                                                    stream);
        break;
      case cuvs::distance::DistanceType::L2SqrtExpanded:
      case cuvs::distance::DistanceType::L2Expanded:
        fusedL2NNImpl<DataT, OutT, IdxT, P, decltype(cutlass_red_op), KVPReduceOpT>(nearest_idx,
                                                                                    nearest_dist,
                                                                                    x,
                                                                                    y,
                                                                                    legacy_xn,
                                                                                    legacy_yn,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    workspace,
                                                                                    cutlass_red_op,
                                                                                    pairRedOp,
                                                                                    sqrt,
                                                                                    false,
                                                                                    out,
                                                                                    stream);
        break;
      default: assert("only cosine/l2 metric is supported with fusedDistanceNN\n"); break;
    }
  };

  MinAndDistanceReduceOpImpl<IdxT, DataT> cutlass_red_op;
  if (cutlass_kvp_scratch != nullptr) {
    if (initOutBuffer) { initFused1nnOutput(nearest_idx, nearest_dist, m, maxVal, stream); }
    cutlass_red_op.out_kvp = cutlass_kvp_scratch;
    initialize<DataT, KVP, IdxT, decltype(cutlass_red_op)>(
      cutlass_kvp_scratch, m, maxVal, cutlass_red_op, stream);
    launch_legacy(cutlass_kvp_scratch, cutlass_red_op);
    if (nearest_idx != nullptr || nearest_dist != nullptr) {
      unpackFused1nnKvpToSoa(nearest_idx, nearest_dist, cutlass_kvp_scratch, m, stream);
    }
  } else {
    RAFT_EXPECTS(nearest_idx == nullptr && nearest_dist != nullptr,
                 "Direct CUTLASS output supports distance-only results");
    cutlass_red_op.out_dist = nearest_dist;
    initialize<DataT, DataT, IdxT, decltype(cutlass_red_op)>(
      nearest_dist, m, maxVal, cutlass_red_op, stream);
    launch_legacy(nearest_dist, cutlass_red_op);
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
