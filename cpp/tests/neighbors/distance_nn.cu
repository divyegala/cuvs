/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_nn_helper.cuh"

#include "../../src/distance/fused_distance_nn.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

#include <cuvs/detail/jit_lto/cutile_module.hpp>
#include <cuvs/detail/jit_lto/tileir_compat.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>

#include <algorithm>

namespace cuvs::neighbors {

enum class ImplType { fused, unfused };

template <typename IdxT>
struct NNInputs {
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  uint64_t rng_seed;
  double tol;
};

__global__ void fill_int8(int8_t* buff, int len, int seed_offset)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Fill the buffer with pseudo-random int8_t values using a simple LCG
  for (int i = tid; i < len; i += blockDim.x * gridDim.x) {
    int hash = (i + seed_offset) * 1103515245 + 12345;
    buff[i]  = static_cast<int8_t>((hash >> 16) & 0xFF);
  }
}

template <typename DataT, typename AccT, typename IdxT, ImplType impl>
class NNTest : public ::testing::TestWithParam<NNInputs<IdxT>> {
 public:
  using RefOutT = raft::KeyValuePair<IdxT, AccT>;
  NNTest()
    : params_{::testing::TestWithParam<NNInputs<IdxT>>::GetParam()},
      m{params_.m},
      n{params_.n},
      k{params_.k},
      metric{params_.metric},
      sqrt{params_.sqrt},
      stream{raft::resource::get_cuda_stream(handle)},
      x{raft::make_device_matrix<DataT, IdxT>(handle, m, k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, n, k)},
      x_norm{raft::make_device_vector<AccT, IdxT>(handle, m)},
      y_norm{raft::make_device_vector<AccT, IdxT>(handle, n)},
      out_idx{raft::make_device_vector<IdxT, IdxT>(handle, m)},
      out_dist{raft::make_device_vector<AccT, IdxT>(handle, m)},
      out_kvp{raft::make_device_vector<RefOutT, IdxT>(handle, m)},
      ref_out{raft::make_device_vector<RefOutT, IdxT>(handle, m)}
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState rng{params_.rng_seed};
    if constexpr (std::is_same_v<DataT, int8_t>) {
      fill_int8<<<1000, 256, 0, stream>>>(x.data_handle(), m * k, 0);
      RAFT_CUDA_TRY(cudaGetLastError());
      fill_int8<<<1000, 256, 0, stream>>>(y.data_handle(), n * k, m * k);
      RAFT_CUDA_TRY(cudaGetLastError());
    } else {
      raft::random::uniform(handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
      raft::random::uniform(handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));
    }

    // Pre-compute norms
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm.data_handle(), x.data_handle(), k, m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      y_norm.data_handle(), y.data_handle(), k, n, stream);

    // CosineExpanded expects ||x|| not ||x||^2
    if (metric == DistanceType::CosineExpanded) {
      raft::linalg::unaryOp(x_norm.data_handle(), x_norm.data_handle(), m, raft::sqrt_op{}, stream);
      raft::linalg::unaryOp(y_norm.data_handle(), y_norm.data_handle(), n, raft::sqrt_op{}, stream);
    }

    if constexpr (impl == ImplType::fused) {
      workspace_size = m * sizeof(IdxT);
    } else if constexpr (impl == ImplType::unfused) {
      workspace_size = m * n * sizeof(AccT);
    }

    raft::matrix::fill(handle, raft::make_device_matrix_view(out_idx.data_handle(), m, 1), IdxT{0});
    raft::matrix::fill(
      handle, raft::make_device_matrix_view(out_dist.data_handle(), m, 1), AccT{0});
    raft::matrix::fill(
      handle, raft::make_device_matrix_view(ref_out.data_handle(), m, 1), RefOutT{0, 0});
    raft::resource::sync_stream(handle, stream);
  }

  void compute_1nn()
  {
    raft::device_vector<char, IdxT> workspace =
      raft::make_device_vector<char, IdxT>(handle, workspace_size);

    ref_nn<DataT, AccT, RefOutT, IdxT>(
      ref_out.data_handle(), x.data_handle(), y.data_handle(), m, n, k, sqrt, metric, stream);

    if constexpr (impl == ImplType::fused) {
      if constexpr (std::is_same_v<DataT, float>) {
        cuvs::distance::fusedDistanceNNMinReduce<DataT, IdxT>(out_idx.data_handle(),
                                                              out_dist.data_handle(),
                                                              x.data_handle(),
                                                              y.data_handle(),
                                                              x_norm.data_handle(),
                                                              y_norm.data_handle(),
                                                              m,
                                                              n,
                                                              k,
                                                              (void*)workspace.data_handle(),
                                                              sqrt,
                                                              false,
                                                              true,
                                                              metric,
                                                              0.0,
                                                              out_kvp.data_handle(),
                                                              stream);
      } else {
        static_assert(sizeof(DataT) == 0,
                      "fusedDistanceNNMinReduce is not implemented for datatype other than float");
      }
    } else if constexpr (impl == ImplType::unfused) {
      cuvs::distance::unfusedDistanceNNMinReduce<DataT, AccT, RefOutT, IdxT>(
        handle,
        out_kvp.data_handle(),
        x.data_handle(),
        y.data_handle(),
        x_norm.data_handle(),
        y_norm.data_handle(),
        m,
        n,
        k,
        (AccT*)workspace.data_handle(),
        sqrt,
        true,
        true,
        metric,
        0.0,
        stream);
    }
  }

  void compare()
  {
    if constexpr (impl == ImplType::fused) {
      vector_compare_soa(
        handle, ref_out.data_handle(), out_idx.data_handle(), out_dist.data_handle(), m, summary);
      // FP32 Tensor Core inputs are rounded to TF32, so near-tied candidates may select a
      // different valid nearest neighbor than the full-FP32 reference.
      const auto allowed_misses = std::max<IdxT>(1, (m + 499) / 500);
      ASSERT_LE(summary.n_misses, allowed_misses) << summary;
    } else {
      vector_compare(handle, ref_out.data_handle(), out_kvp.data_handle(), m, summary);
    }
    ASSERT_TRUE(summary.max_diff < params_.tol) << summary;
  }

 private:
  raft::resources handle;
  rmm::cuda_stream_view stream;
  NNInputs<IdxT> params_;
  ComparisonSummary summary;
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  raft::device_matrix<DataT, IdxT> x;
  raft::device_matrix<DataT, IdxT> y;
  raft::device_vector<AccT, IdxT> x_norm;
  raft::device_vector<AccT, IdxT> y_norm;
  raft::device_vector<IdxT, IdxT> out_idx;
  raft::device_vector<AccT, IdxT> out_dist;
  raft::device_vector<RefOutT, IdxT> out_kvp;
  raft::device_vector<RefOutT, IdxT> ref_out;
  size_t workspace_size;
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {16384, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {8192, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  // Fused implementation for cosine distance ignores the sqrt parameter, therefore
  // commenting the following two tests
  // {4096, 4096, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
  // {4096, 8192, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32_fused = [] {
  auto inputs = input_fp32<IdxT>;
#if CUVS_CUTILE_ENABLED
  inputs.insert(
    inputs.begin() + 6,
    NNInputs<IdxT>{512, 1024, 64, DistanceType::InnerProduct, false, uint64_t(31415926), 0.1});
#endif
  inputs.push_back(
    NNInputs<IdxT>{1000, 8, 32, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1});
  inputs.push_back(
    NNInputs<IdxT>{1000, 40, 16, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1});
  return inputs;
}();

// Test fused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::fused> NNTest_fp32_fused;
TEST_P(NNTest_fp32_fused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_fused, ::testing::ValuesIn(input_fp32_fused<int>));

// Test unfused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::unfused> NNTest_fp32_unfused;
TEST_P(NNTest_fp32_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_unfused, ::testing::ValuesIn(input_fp32<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp16 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// Test unfused implementation with fp16, int8
// Fused implementation has no support for fp16, int8 so no test for it
typedef NNTest<half, float, int32_t, ImplType::unfused> NNTest_fp16_unfused;
TEST_P(NNTest_fp16_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp16_unfused, ::testing::ValuesIn(input_fp16<int>));

TEST(Fused1nn, ExpandedL2ClampsNegativeRoundoff)
{
  raft::resources handle;
  auto stream     = raft::resource::get_cuda_stream(handle);
  constexpr int k = 64;

  auto x         = raft::make_device_matrix<float, int>(handle, 1, k);
  auto y         = raft::make_device_matrix<float, int>(handle, 1, k);
  auto x_norm    = raft::make_device_vector<float, int>(handle, 1);
  auto y_norm    = raft::make_device_vector<float, int>(handle, 1);
  auto out_idx   = raft::make_device_vector<int, int>(handle, 1);
  auto out_dist  = raft::make_device_vector<float, int>(handle, 1);
  auto out_kvp   = raft::make_device_vector<raft::KeyValuePair<int, float>, int>(handle, 1);
  auto workspace = raft::make_device_vector<int, int>(handle, 1);

  raft::matrix::fill(handle, x.view(), 1.0006f);
  raft::copy(y.data_handle(), x.data_handle(), k, stream);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    x_norm.data_handle(), x.data_handle(), k, 1, stream);
  raft::copy(y_norm.data_handle(), x_norm.data_handle(), 1, stream);

  cuvs::distance::fusedDistanceNNMinReduce<float, int>(out_idx.data_handle(),
                                                       out_dist.data_handle(),
                                                       x.data_handle(),
                                                       y.data_handle(),
                                                       x_norm.data_handle(),
                                                       y_norm.data_handle(),
                                                       1,
                                                       1,
                                                       k,
                                                       workspace.data_handle(),
                                                       true,
                                                       true,
                                                       true,
                                                       DistanceType::L2SqrtExpanded,
                                                       0.0f,
                                                       out_kvp.data_handle(),
                                                       stream);

  int actual_idx;
  float actual_dist;
  raft::update_host(&actual_idx, out_idx.data_handle(), 1, stream);
  raft::update_host(&actual_dist, out_dist.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(actual_idx, 0);
  EXPECT_EQ(actual_dist, 0.0f);
}

TEST(Fused1nn, CutileAvailabilityRejectsUnsupportedArchitectures)
{
  EXPECT_FALSE(cuvs::detail::jit_lto::cutile_launch_available_for_arch(7, 5, 13000));
  EXPECT_FALSE(cuvs::detail::jit_lto::cutile_launch_available_for_arch(13, 0, 13000));
}

#if CUVS_CUTILE_ENABLED
TEST(Fused1nn, ExpectedModuleCompatibilityErrorsAreRecoverable)
{
  using cuvs::detail::jit_lto::is_expected_cutile_unavailable;
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorInvalidDeviceFunction));
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorInvalidPtx));
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorNoKernelImageForDevice));
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorSymbolNotFound));
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorUnsupportedPtxVersion));
  EXPECT_TRUE(is_expected_cutile_unavailable(cudaErrorCallRequiresNewerDriver));
  EXPECT_FALSE(is_expected_cutile_unavailable(cudaErrorMemoryAllocation));
  EXPECT_FALSE(is_expected_cutile_unavailable(cudaErrorIllegalAddress));
}

template <typename IdxT>
void run_half_cutile_contract_case(int k)
{
  raft::resources handle;
  auto stream      = raft::resource::get_cuda_stream(handle);
  constexpr IdxT m = 2;
  constexpr IdxT n = 2;

  std::vector<half> h_x(static_cast<size_t>(m) * k, __float2half(0.0f));
  std::vector<half> h_y(static_cast<size_t>(n) * k, __float2half(0.0f));
  h_x[0]     = __float2half(1.0f);
  h_x[k + 1] = __float2half(1.0f);
  h_y[0]     = __float2half(1.0f);
  h_y[k + 1] = __float2half(1.0f);

  rmm::device_uvector<half> x(h_x.size(), stream);
  rmm::device_uvector<half> y(h_y.size(), stream);
  rmm::device_uvector<float> x_norm(m, stream);
  rmm::device_uvector<float> y_norm(n, stream);
  rmm::device_uvector<IdxT> out_idx(m, stream);
  rmm::device_uvector<half> out_dist(m, stream);
  rmm::device_uvector<int> workspace(m, stream);
  raft::update_device(x.data(), h_x.data(), h_x.size(), stream);
  raft::update_device(y.data(), h_y.data(), h_y.size(), stream);
  const std::vector<float> h_norms(m, 1.0f);
  raft::update_device(x_norm.data(), h_norms.data(), m, stream);
  raft::update_device(y_norm.data(), h_norms.data(), n, stream);

  if constexpr (std::is_same_v<IdxT, int64_t>) {
    EXPECT_FALSE((cuvs::distance::detail::try_fused_1nn_tile<half, IdxT>(out_idx.data(),
                                                                         out_dist.data(),
                                                                         x.data(),
                                                                         y.data(),
                                                                         x_norm.data(),
                                                                         y_norm.data(),
                                                                         m,
                                                                         n,
                                                                         static_cast<IdxT>(k),
                                                                         DistanceType::L2Expanded,
                                                                         false,
                                                                         x.data(),
                                                                         stream)));
  }

  for (auto metric : {DistanceType::L2Expanded,
                      DistanceType::L2SqrtExpanded,
                      DistanceType::CosineExpanded,
                      DistanceType::InnerProduct}) {
    ASSERT_TRUE((
      cuvs::distance::detail::try_fused_1nn_tile<half, IdxT>(out_idx.data(),
                                                             out_dist.data(),
                                                             x.data(),
                                                             y.data(),
                                                             x_norm.data(),
                                                             y_norm.data(),
                                                             m,
                                                             n,
                                                             static_cast<IdxT>(k),
                                                             metric,
                                                             metric == DistanceType::L2SqrtExpanded,
                                                             workspace.data(),
                                                             stream)));
    std::vector<IdxT> h_idx(m);
    raft::update_host(h_idx.data(), out_idx.data(), m, stream);
    raft::resource::sync_stream(handle, stream);
    EXPECT_EQ(h_idx[0], IdxT{0});
    EXPECT_EQ(h_idx[1], IdxT{1});
  }

  ASSERT_TRUE((cuvs::distance::detail::try_fused_1nn_tile<half, IdxT>(out_idx.data(),
                                                                      out_dist.data(),
                                                                      x.data(),
                                                                      x.data(),
                                                                      x_norm.data(),
                                                                      x_norm.data(),
                                                                      m,
                                                                      m,
                                                                      static_cast<IdxT>(k),
                                                                      DistanceType::L2Expanded,
                                                                      false,
                                                                      workspace.data(),
                                                                      stream)));
  std::vector<IdxT> h_alias_idx(m);
  raft::update_host(h_alias_idx.data(), out_idx.data(), m, stream);
  raft::resource::sync_stream(handle, stream);
  EXPECT_EQ(h_alias_idx[0], IdxT{0});
  EXPECT_EQ(h_alias_idx[1], IdxT{1});

  raft::copy(y.data() + k, y.data(), k, stream);
  ASSERT_TRUE((cuvs::distance::detail::try_fused_1nn_tile<half, IdxT>(out_idx.data(),
                                                                      out_dist.data(),
                                                                      x.data(),
                                                                      y.data(),
                                                                      x_norm.data(),
                                                                      y_norm.data(),
                                                                      m,
                                                                      n,
                                                                      static_cast<IdxT>(k),
                                                                      DistanceType::L2Expanded,
                                                                      false,
                                                                      workspace.data(),
                                                                      stream)));
  IdxT h_tie_idx;
  half h_tie_dist;
  raft::update_host(&h_tie_idx, out_idx.data(), 1, stream);
  raft::update_host(&h_tie_dist, out_dist.data(), 1, stream);
  raft::resource::sync_stream(handle, stream);
  EXPECT_TRUE(h_tie_idx == IdxT{0} || h_tie_idx == IdxT{1});
  EXPECT_EQ(__half2float(h_tie_dist), 0.0f);
}

TEST(Fused1nn, HalfUsesFloatNormsAcrossAbisAndIndexTypes)
{
  run_half_cutile_contract_case<int>(8);
  run_half_cutile_contract_case<int>(7);
  run_half_cutile_contract_case<int64_t>(8);
  run_half_cutile_contract_case<int64_t>(7);
}
#endif

TEST(Fused1nn, Int64IndexWorkspaceUsesLargestChunk)
{
  constexpr int64_t max_batch_m_float = cuvs::distance::detail::fused_1nn_cutile_max_batch_m<float>;
  constexpr int64_t max_batch_m_half  = cuvs::distance::detail::fused_1nn_cutile_max_batch_m<half>;
  EXPECT_EQ(max_batch_m_float, 2147483644);
  EXPECT_EQ(max_batch_m_half, 2147483640);
  EXPECT_EQ(cuvs::distance::detail::fused_1nn_cutile_index_workspace_rows<float>(1024), 1024);
  EXPECT_EQ(cuvs::distance::detail::fused_1nn_cutile_index_workspace_rows<float>(
              std::numeric_limits<int64_t>::max()),
            static_cast<size_t>(max_batch_m_float));
  EXPECT_EQ(cuvs::distance::detail::fused_1nn_cutile_index_workspace_rows<half>(
              std::numeric_limits<int64_t>::max()),
            static_cast<size_t>(max_batch_m_half));
}

#if CUVS_CUTILE_ENABLED
TEST(Fused1nn, PointerAwareProbeRejectsMisalignedArrays)
{
  raft::resources handle;
  constexpr int m = 32;
  constexpr int n = 32;
  constexpr int k = 64;

  auto x        = raft::make_device_vector<float, int>(handle, m * k + 1);
  auto y        = raft::make_device_vector<float, int>(handle, n * k);
  auto x_norm   = raft::make_device_vector<float, int>(handle, m + 1);
  auto y_norm   = raft::make_device_vector<float, int>(handle, n);
  auto out_idx  = raft::make_device_vector<int, int>(handle, m);
  auto out_dist = raft::make_device_vector<float, int>(handle, m);

  for (auto metric :
       {DistanceType::L2Expanded, DistanceType::CosineExpanded, DistanceType::InnerProduct}) {
    EXPECT_FALSE(cuvs::distance::detail::can_launch_fused_1nn_tile(out_idx.data_handle(),
                                                                   out_dist.data_handle(),
                                                                   x.data_handle() + 1,
                                                                   y.data_handle(),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   metric));

    if (cuvs::distance::detail::can_launch_fused_1nn_tile(out_idx.data_handle(),
                                                          out_dist.data_handle(),
                                                          x.data_handle(),
                                                          y.data_handle(),
                                                          m,
                                                          n,
                                                          k,
                                                          metric)) {
      EXPECT_FALSE(cuvs::distance::detail::can_launch_fused_1nn_tile(out_idx.data_handle(),
                                                                     out_dist.data_handle(),
                                                                     x.data_handle(),
                                                                     y.data_handle(),
                                                                     x_norm.data_handle() + 1,
                                                                     y_norm.data_handle(),
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     metric));
      EXPECT_FALSE(cuvs::distance::detail::can_launch_fused_1nn_tile(
        out_idx.data_handle(), x.data_handle(), x.data_handle(), y.data_handle(), m, n, k, metric));
    }
  }
}
#endif

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_int8 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// DataT = int8_t, AccT = int32_t
typedef NNTest<int8_t, int32_t, int32_t, ImplType::unfused> NNTest_int8_unfused;
TEST_P(NNTest_int8_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused, ::testing::ValuesIn(input_int8<int>));

// DataT = int8_t, AccT = float
typedef NNTest<int8_t, float, int32_t, ImplType::unfused> NNTest_int8_unfused2;
TEST_P(NNTest_int8_unfused2, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused2, ::testing::ValuesIn(input_int8<int>));
}  // namespace cuvs::neighbors
