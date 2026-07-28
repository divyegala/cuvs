/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/core/logger.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuvs::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<ViewT>
void serialize(const raft::resources& res, std::ostream& os, ViewT const& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  auto src = dataset.view();
  auto dst = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  if constexpr (cuvs::neighbors::is_device_dataset_view_v<ViewT>) {
    raft::copy_matrix(dst.data_handle(),
                      dim,
                      src.data_handle(),
                      stride,
                      dim,
                      n_rows,
                      raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  } else {
    for (IdxT row = 0; row < n_rows; ++row) {
      std::memcpy(
        dst.data_handle() + row * dim, src.data_handle() + row * stride, dim * sizeof(DataT));
    }
  }
  raft::serialize_mdspan(res, os, dst.view());
}

/** Write CAGRA index dataset blob (tag + element dtype + strided payload). */
template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<ViewT>
void serialize_cagra_dense_dataset(const raft::resources& res,
                                   std::ostream& os,
                                   ViewT const& dataset)
{
  raft::serialize_scalar(res, os, kSerializeStridedDataset);
  if constexpr (std::is_same_v<DataT, float>) {
    raft::serialize_scalar(res, os, CUDA_R_32F);
  } else if constexpr (std::is_same_v<DataT, half>) {
    raft::serialize_scalar(res, os, CUDA_R_16F);
  } else if constexpr (std::is_same_v<DataT, int8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8I);
  } else if constexpr (std::is_same_v<DataT, uint8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8U);
  } else {
    static_assert(!std::is_same_v<DataT, DataT>, "unsupported element type for CAGRA serialize");
  }
  serialize<DataT, IdxT>(res, os, dataset);
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_empty_dataset<IdxT>>
{
  auto suggested_dim = raft::deserialize_scalar<uint32_t>(res, is);
  return std::make_unique<device_empty_dataset<IdxT>>(suggested_dim);
}

/** Read shared dense wire payload: `n_rows`, `dim`, `stride`, then tight `[n_rows x dim]` host
 * matrix. */
template <typename DataT, typename IdxT>
auto deserialize_dense_payload(raft::resources const& res, std::istream& is)
  -> std::tuple<IdxT, uint32_t, uint32_t, raft::host_matrix<DataT, IdxT>>
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "deserialize_dense_payload: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  return {n_rows, dim, stride, std::move(host_array)};
}

template <typename DataT, typename IdxT>
void skip_dense_payload(raft::resources const& res, std::istream& is)
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "skip_dense_payload: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  if constexpr (std::is_signed_v<IdxT>) {
    RAFT_EXPECTS(n_rows >= 0, "skip_dense_payload: row count must not be negative");
  }

  auto const header         = raft::numpy_serializer::read_header(is);
  auto const expected_dtype = raft::numpy_serializer::get_numpy_dtype<DataT>();
  RAFT_EXPECTS(header.dtype == expected_dtype,
               "skip_dense_payload: expected dtype %s but got %s",
               expected_dtype.to_string().c_str(),
               header.dtype.to_string().c_str());
  RAFT_EXPECTS(!header.fortran_order, "skip_dense_payload: expected row-major payload");
  RAFT_EXPECTS(header.shape.size() == 2,
               "skip_dense_payload: expected rank 2 but got %zu",
               header.shape.size());

  using payload_size_t = raft::numpy_serializer::ndarray_len_t;
  auto const rows      = static_cast<payload_size_t>(n_rows);
  auto const columns   = static_cast<payload_size_t>(dim);
  RAFT_EXPECTS(header.shape[0] == rows && header.shape[1] == columns,
               "skip_dense_payload: payload shape does not match serialized dimensions");
  RAFT_EXPECTS(rows == 0 || columns <= std::numeric_limits<payload_size_t>::max() / rows,
               "skip_dense_payload: element count overflow");
  auto const elements = rows * columns;
  RAFT_EXPECTS(elements <= std::numeric_limits<payload_size_t>::max() / sizeof(DataT),
               "skip_dense_payload: byte count overflow");
  auto remaining = elements * sizeof(DataT);

  using pos_type              = std::istream::pos_type;
  using off_type              = std::istream::off_type;
  auto* buffer                = is.rdbuf();
  auto const invalid_position = pos_type{off_type{-1}};
  auto const current          = buffer->pubseekoff(0, std::ios_base::cur, std::ios_base::in);
  if (current != invalid_position &&
      remaining <= static_cast<payload_size_t>(std::numeric_limits<off_type>::max())) {
    auto const end = buffer->pubseekoff(0, std::ios_base::end, std::ios_base::in);
    if (end != invalid_position) {
      auto const available = end - current;
      RAFT_EXPECTS(available >= 0 && static_cast<payload_size_t>(available) >= remaining,
                   "skip_dense_payload: truncated payload");
      auto const next =
        buffer->pubseekpos(current + static_cast<off_type>(remaining), std::ios_base::in);
      RAFT_EXPECTS(next != invalid_position, "skip_dense_payload: failed to seek past payload");
      return;
    }
    RAFT_EXPECTS(buffer->pubseekpos(current, std::ios_base::in) != invalid_position,
                 "skip_dense_payload: failed to restore stream position");
  }

  std::array<char, 64 * 1024> discard_buffer{};
  while (remaining > 0) {
    auto const chunk = std::min<payload_size_t>(remaining, discard_buffer.size());
    is.read(discard_buffer.data(), static_cast<std::streamsize>(chunk));
    RAFT_EXPECTS(static_cast<payload_size_t>(is.gcount()) == chunk,
                 "skip_dense_payload: truncated payload");
    remaining -= chunk;
  }
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_device_dense(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  auto payload = deserialize_dense_payload<DataT, IdxT>(res, is);
  return make_device_dense_row_major_dataset_from_src<OwningDatasetT, DataT, IdxT>(
    res,
    std::get<3>(payload).view(),
    std::get<1>(payload),
    std::get<2>(payload),
    "deserialize_dense_dataset");
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_host_dense(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  auto payload = deserialize_dense_payload<DataT, IdxT>(res, is);
  auto n_rows  = std::get<0>(payload);
  auto dim     = std::get<1>(payload);
  auto stride  = std::get<2>(payload);
  auto src     = std::get<3>(payload).view();
  auto storage = raft::make_host_matrix<DataT, IdxT>(n_rows, stride);
  std::memset(storage.data_handle(), 0, storage.size() * sizeof(DataT));
  for (IdxT row = 0; row < n_rows; ++row) {
    std::memcpy(
      storage.data_handle() + row * stride, src.data_handle() + row * dim, dim * sizeof(DataT));
  }
  return std::make_unique<OwningDatasetT>(std::move(storage), dim);
}

template <typename DataT, typename IdxT>
auto deserialize_vpq(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_vpq_dataset<DataT, IdxT>>
{
  auto n_rows             = raft::deserialize_scalar<IdxT>(res, is);
  auto dim                = raft::deserialize_scalar<uint32_t>(res, is);
  auto vq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_len             = raft::deserialize_scalar<uint32_t>(res, is);
  auto encoded_row_length = raft::deserialize_scalar<uint32_t>(res, is);

  auto vq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, vq_n_centers, dim);
  auto pq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto data =
    raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, encoded_row_length);

  raft::deserialize_mdspan(res, is, vq_code_book.view());
  raft::deserialize_mdspan(res, is, pq_code_book.view());
  raft::deserialize_mdspan(res, is, data.view());

  return std::make_unique<device_vpq_dataset<DataT, IdxT>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(data));
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_dense_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  RAFT_EXPECTS(tag == kSerializeStridedDataset,
               "deserialize_dataset: expected strided tag, got %u",
               static_cast<unsigned>(tag));
  const auto dtype                        = raft::deserialize_scalar<cudaDataType_t>(res, is);
  constexpr cudaDataType_t expected_dtype = std::is_same_v<DataT, float>    ? CUDA_R_32F
                                            : std::is_same_v<DataT, half>   ? CUDA_R_16F
                                            : std::is_same_v<DataT, int8_t> ? CUDA_R_8I
                                                                            : CUDA_R_8U;  // uint8_t
  RAFT_EXPECTS(dtype == expected_dtype,
               "deserialize_dataset: serialized dtype (%d) does not match expected (%d)",
               static_cast<int>(dtype),
               static_cast<int>(expected_dtype));
  if constexpr (std::is_same_v<OwningDatasetT, device_padded_dataset<DataT, IdxT>>) {
    return deserialize_device_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else if constexpr (std::is_same_v<OwningDatasetT, device_standard_dataset<DataT, IdxT>>) {
    return deserialize_device_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else if constexpr (std::is_same_v<OwningDatasetT, host_padded_dataset<DataT, IdxT>>) {
    return deserialize_host_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else if constexpr (std::is_same_v<OwningDatasetT, host_standard_dataset<DataT, IdxT>>) {
    return deserialize_host_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else {
    static_assert(!std::is_same_v<OwningDatasetT, OwningDatasetT>,
                  "deserialize_dense_dataset: unsupported owning dataset type");
  }
}

template <typename DataT, typename IdxT>
void skip_dense_dataset(raft::resources const& res, std::istream& is)
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  RAFT_EXPECTS(tag == kSerializeStridedDataset,
               "skip_dense_dataset: expected strided tag, got %u",
               static_cast<unsigned>(tag));
  const auto dtype                        = raft::deserialize_scalar<cudaDataType_t>(res, is);
  constexpr cudaDataType_t expected_dtype = std::is_same_v<DataT, float>    ? CUDA_R_32F
                                            : std::is_same_v<DataT, half>   ? CUDA_R_16F
                                            : std::is_same_v<DataT, int8_t> ? CUDA_R_8I
                                                                            : CUDA_R_8U;
  RAFT_EXPECTS(dtype == expected_dtype,
               "skip_dense_dataset: serialized dtype (%d) does not match expected (%d)",
               static_cast<int>(dtype),
               static_cast<int>(expected_dtype));
  skip_dense_payload<DataT, IdxT>(res, is);
}

// Reads tag + dtype prefix, validates they match DataT, and returns the requested concrete
// dense owner. When a new dataset kind is supported, add a matching overload of
// deserialize_dataset here rather than extending this one — overload dispatch replaces the old
// type-erased variant routing.
template <typename DataT, typename IdxT>
auto deserialize_padded_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_padded_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_padded_dataset<DataT, IdxT>>(res, is);
}

template <typename DataT, typename IdxT>
auto deserialize_standard_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_standard_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_standard_dataset<DataT, IdxT>>(res, is);
}

template <typename DataT, typename IdxT>
auto deserialize_host_padded_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<host_padded_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, host_padded_dataset<DataT, IdxT>>(res, is);
}

template <typename DataT, typename IdxT>
auto deserialize_host_standard_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<host_standard_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, host_standard_dataset<DataT, IdxT>>(res, is);
}

}  // namespace cuvs::neighbors::detail
