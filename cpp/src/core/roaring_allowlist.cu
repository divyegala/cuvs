/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>

#include "nvtx.hpp"

#include <cuco/roaring_bitmap.cuh>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/std/cstddef>
#include <cuda/stream_ref>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace cuvs::core {
namespace {

/**
 * cuCollections Roaring ownership and encoding
 * =============================================
 *
 * cuCollections constructs a standard portable 32-bit Roaring stream directly on the GPU. A
 * 32-bit ID is divided into a high-16-bit container key and a low-16-bit value. For each key, at
 * most 4,096 values are encoded as a sorted uint16 array; larger containers use an 8 KiB bitmap.
 * ID construction does not emit run containers.
 *
 * The owning cuco object retains the encoded bytes. Its lightweight
 * cuco::experimental::roaring_bitmap_ref<uint32_t> parses the portable header once and then stores
 * container-location metadata plus pointers into those bytes. cuVS materializes that reference in
 * a stable device allocation during construction. Views and CAGRA filters copy only its pointer, so
 * search never copies or reparses the payload.
 *
 * @see https://github.com/RoaringBitmap/RoaringFormatSpec
 * @see https://github.com/NVIDIA/cuCollections/pull/839
 */

using ref_type              = cuco::experimental::roaring_bitmap_ref<std::uint32_t>;
using cuco_bitmap_allocator = rmm::mr::polymorphic_allocator<cuda::std::byte>;
using cuco_bitmap_type = cuco::experimental::roaring_bitmap<std::uint32_t, cuco_bitmap_allocator>;

constexpr int kBlockSize = 256;

int grid_size_for(std::size_t count)
{
  auto const blocks = (count + kBlockSize - 1) / kBlockSize;
  return static_cast<int>(std::min<std::size_t>(blocks, 65535));
}

void validate_dataset_rows(std::size_t dataset_rows)
{
  constexpr std::uint64_t kKeyDomain = std::uint64_t{1} << 32;
  RAFT_EXPECTS(dataset_rows > 0, "dataset_rows must be greater than zero.");
  RAFT_EXPECTS(static_cast<std::uint64_t>(dataset_rows) <= kKeyDomain,
               "dataset_rows exceeds the uint32_t Roaring key domain.");
}

__global__ void validate_input_kernel(std::uint32_t const* ids,
                                      std::size_t size,
                                      std::uint64_t dataset_rows,
                                      std::uint32_t* invalid)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size && static_cast<std::uint64_t>(ids[i]) >= dataset_rows) { atomicExch(invalid, 1u); }
}

__global__ void store_ref_kernel(ref_type ref, ref_type* output)
{
  if (threadIdx.x == 0) { ::new (static_cast<void*>(output)) ref_type{ref}; }
}

__global__ void contains_kernel(ref_type const* reference,
                                bool empty,
                                std::uint64_t dataset_rows,
                                std::uint32_t const* row_ids,
                                std::uint8_t* output,
                                std::size_t size)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size) { return; }
  auto const row = row_ids[i];
  output[i] = !empty && static_cast<std::uint64_t>(row) < dataset_rows && reference->contains(row);
}

struct device_build_result {
  std::unique_ptr<cuco_bitmap_type> owner;
  rmm::device_uvector<cuda::std::byte> reference;
  std::size_t cardinality{};
};

device_build_result build_from_device_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  bool pre_sorted)
{
  common::nvtx::range<common::nvtx::domain::cuvs> build_scope("roaring_allowlist::build_from_ids");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  if (size == 0) { return {nullptr, rmm::device_uvector<cuda::std::byte>(0, stream), 0}; }

  rmm::device_uvector<std::uint32_t> invalid(1, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(invalid.data(), 0, sizeof(std::uint32_t), stream));
  validate_input_kernel<<<grid_size_for(size), kBlockSize, 0, stream>>>(
    ids.data_handle(), size, dataset_rows, invalid.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::uint32_t host_invalid{};
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &host_invalid, invalid.data(), sizeof(host_invalid), cudaMemcpyDeviceToHost, stream));

  cuco_bitmap_allocator allocator{};
  cuda::stream_ref cuco_stream{stream.value()};
  auto bitmap = pre_sorted ? cuco_bitmap_type::from_sorted_unique_indices(
                               ids.data_handle(), ids.data_handle() + size, allocator, cuco_stream)
                           : cuco_bitmap_type::from_indices(
                               ids.data_handle(), ids.data_handle() + size, allocator, cuco_stream);

  // The exact-size readback in the cuco factory orders the preceding validation copy.
  RAFT_EXPECTS(host_invalid == 0, "Roaring allowlist ID must be smaller than dataset_rows.");

  auto owner             = std::make_unique<cuco_bitmap_type>(std::move(bitmap));
  auto const cardinality = static_cast<std::size_t>(owner->size());
  rmm::device_uvector<cuda::std::byte> reference(sizeof(ref_type), stream);
  store_ref_kernel<<<1, 1, 0, stream>>>(owner->ref(),
                                        reinterpret_cast<ref_type*>(reference.data()));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return {std::move(owner), std::move(reference), cardinality};
}

}  // namespace

struct roaring_allowlist::impl {
  std::unique_ptr<cuco_bitmap_type> owner;
  rmm::device_uvector<cuda::std::byte> reference_storage;
  std::size_t cardinality_{};
  std::size_t dataset_rows_{};

  impl(std::size_t dataset_rows, device_build_result&& built)
    : owner(std::move(built.owner)),
      reference_storage(std::move(built.reference)),
      cardinality_(built.cardinality),
      dataset_rows_(dataset_rows)
  {
    static_assert(std::is_trivially_destructible_v<ref_type>);
  }

  [[nodiscard]] ref_type const* reference() const noexcept
  {
    return cardinality_ == 0 ? nullptr
                             : reinterpret_cast<ref_type const*>(reference_storage.data());
  }
};

roaring_allowlist::roaring_allowlist(std::unique_ptr<impl> impl) noexcept : impl_(std::move(impl))
{
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::host_vector_view<const key_type, std::int64_t> ids,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  auto const stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<key_type> device_ids(size, stream);
  if (size != 0) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_ids.data(),
                                  ids.data_handle(),
                                  size * sizeof(key_type),
                                  cudaMemcpyHostToDevice,
                                  stream));
  }
  auto device_ids_view = raft::make_device_vector_view<const key_type, std::int64_t>(
    device_ids.data(), static_cast<std::int64_t>(size));
  auto built = build_from_device_ids(res, dataset_rows, device_ids_view, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(dataset_rows, std::move(built))};
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const key_type, std::int64_t> ids,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  auto built = build_from_device_ids(res, dataset_rows, ids, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(dataset_rows, std::move(built))};
}

roaring_allowlist::~roaring_allowlist()                                       = default;
roaring_allowlist::roaring_allowlist(roaring_allowlist&&) noexcept            = default;
roaring_allowlist& roaring_allowlist::operator=(roaring_allowlist&&) noexcept = default;

std::size_t roaring_allowlist::dataset_rows() const noexcept { return impl_->dataset_rows_; }

std::size_t roaring_allowlist::cardinality() const noexcept { return impl_->cardinality_; }

bool roaring_allowlist::empty() const noexcept { return cardinality() == 0; }

std::size_t roaring_allowlist::size_bytes() const noexcept
{
  auto bytes = impl_->reference_storage.size() * sizeof(cuda::std::byte);
  if (impl_->owner) { bytes += static_cast<std::size_t>(impl_->owner->size_bytes()); }
  return bytes;
}

roaring_allowlist_view roaring_allowlist::view() const noexcept
{
  return roaring_allowlist_view{impl_->reference(), dataset_rows(), cardinality()};
}

void roaring_allowlist::contains(raft::resources const& res,
                                 raft::device_vector_view<const key_type, std::int64_t> row_ids,
                                 raft::device_vector_view<std::uint8_t, std::int64_t> output) const
{
  contains_async(res, row_ids, output);
  raft::resource::sync_stream(res);
}

void roaring_allowlist::contains_async(
  raft::resources const& res,
  raft::device_vector_view<const key_type, std::int64_t> row_ids,
  raft::device_vector_view<std::uint8_t, std::int64_t> output) const
{
  RAFT_EXPECTS(output.extent(0) == row_ids.extent(0),
               "Roaring membership output size must match the input size.");
  auto const size = static_cast<std::size_t>(row_ids.extent(0));
  if (size == 0) { return; }
  contains_kernel<<<grid_size_for(size), kBlockSize, 0, raft::resource::get_cuda_stream(res)>>>(
    impl_->reference(),
    empty(),
    static_cast<std::uint64_t>(dataset_rows()),
    row_ids.data_handle(),
    output.data_handle(),
    size);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::core
