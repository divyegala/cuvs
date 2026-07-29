/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/common.h>

#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic dataset layout kind for C API dataset handles.
 */
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED   = 1
} cuvsDatasetLayout_t;

/**
 * @brief Memory space holding a C API dataset handle's data.
 */
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST   = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;

/**
 * @brief Owning dataset handle.
 *
 * `addr` points to C++ owning dataset storage managed by the C API. `mem_type` identifies the
 * memory space and `layout` identifies the data layout (standard or padded).
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
} cuvsDataset;
typedef cuvsDataset* cuvsDataset_t;

/**
 * @brief Non-owning dataset view handle.
 *
 * `addr` points to C API-owned metadata that references caller-provided tensor memory. The
 * `mem_type` and `layout` fields identify the concrete dataset view type.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
} cuvsDatasetView;
typedef cuvsDatasetView* cuvsDatasetView_t;

/**
 * @brief Create an empty owning dataset handle.
 *
 * The dataset storage, memory type, layout, and dtype are populated by the operation that fills
 * this handle.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset);

/**
 * @brief Create an owning padded dataset from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                              DLManagedTensor* dataset,
                                              cuvsDataset_t* padded_dataset);

/**
 * @brief Create a non-owning padded dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                                  DLManagedTensor* dataset,
                                                  cuvsDatasetView_t* padded_dataset);

/**
 * @brief Create a non-owning standard dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                                    DLManagedTensor* dataset,
                                                    cuvsDatasetView_t* standard_dataset);

/**
 * @brief Create a non-owning view wrapper from an owning dataset.
 *
 * The returned view references the owning dataset's storage. The dataset must outlive the view.
 *
 * @param[in] dataset owning dataset handle
 * @param[out] view output view handle
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeViewWrapper(cuvsDataset_t dataset,
                                                   cuvsDatasetView_t* view);

/** @brief Destroy an owning dataset handle created by a `cuvsDatasetMake*` function. */
CUVS_EXPORT cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset);

/**
 * @brief Destroy a non-owning dataset view handle created by a dataset view factory or
 * `cuvsDatasetMakeViewWrapper`.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetViewDestroy(cuvsDatasetView_t dataset_view);

#ifdef __cplusplus
}
#endif
