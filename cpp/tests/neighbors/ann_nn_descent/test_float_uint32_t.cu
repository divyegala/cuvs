/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

namespace {

const std::vector<AnnNNDescentInputs> directMmaDispatchInputs{
  // Direct TF32 WGMMA uses K=64 tiles. Cover exact and partial tiles across several dimensions.
  {512, 159, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 160, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 161, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 192, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 193, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 256, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::TF32},
  {512, 257, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::TF32},

  // FP16 uses the legacy two-stage kernel. Cover tiny, partial, exact, and large dimensions.
  {512, 7, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 65, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 127, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 128, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 129, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 448, 32, cuvs::distance::DistanceType::InnerProduct, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512, 449, 32, cuvs::distance::DistanceType::CosineExpanded, false, 0.90, DIST_COMP_DTYPE::FP16},
  {512,
   1024,
   32,
   cuvs::distance::DistanceType::CosineExpanded,
   false,
   0.90,
   DIST_COMP_DTYPE::FP16}};

std::string directMmaDispatchName(const ::testing::TestParamInfo<AnnNNDescentInputs>& test_info)
{
  const auto& input      = test_info.param;
  const char* dtype_name = input.dist_comp_dtype == DIST_COMP_DTYPE::TF32 ? "TF32" : "FP16";
  const char* metric_name =
    input.metric == cuvs::distance::DistanceType::InnerProduct ? "InnerProduct" : "Cosine";
  return std::string{dtype_name} + "D" + std::to_string(input.dim) + metric_name;
}

}  // namespace
typedef AnnNNDescentTest<float, float, std::uint32_t> AnnNNDescentTestF_U32;
TEST_P(AnnNNDescentTestF_U32, AnnNNDescent) { this->testNNDescent(); }

typedef AnnNNDescentDistEpiTest<float, float, std::uint32_t> AnnNNDescentTestDistEpiF_U32;
TEST_P(AnnNNDescentTestDistEpiF_U32, AnnNNDescentDistEpi) { this->testNNDescent(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnNNDescentTF32MmaBoundaries,
                        AnnNNDescentTestF_U32,
                        ::testing::ValuesIn(tf32MmaBoundaryInputs));
INSTANTIATE_TEST_CASE_P(AnnNNDescentDirectMmaDispatch,
                        AnnNNDescentTestF_U32,
                        ::testing::ValuesIn(directMmaDispatchInputs),
                        directMmaDispatchName);
INSTANTIATE_TEST_CASE_P(AnnNNDescentDistEpi,
                        AnnNNDescentTestDistEpiF_U32,
                        ::testing::ValuesIn(inputsDistEpilogue));

}  // namespace   cuvs::neighbors::nn_descent
