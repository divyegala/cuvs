/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::marker::PhantomData;

use crate::dlpack::{AsDlTensor, DeviceTypeExt};
use crate::error::check_cuvs;
use crate::ffi_utils::{init_handle, report_drop_failure};
use crate::neighbors::cagra::CagraError;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, CagraError>;

/// Host/device residency and row layout of a [`DatasetView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatasetKind {
    /// Device-resident rows with CAGRA's required padded width.
    DevicePadded,
    /// Device-resident rows with a standard, unpadded width.
    DeviceStandard,
    /// Host-resident rows with CAGRA's required padded width.
    HostPadded,
    /// Host-resident rows with a standard, unpadded width.
    HostStandard,
}

impl DatasetKind {
    fn from_ffi(mem_type: ffi::cuvsDatasetMemType_t, layout: ffi::cuvsDatasetLayout_t) -> Self {
        match (mem_type, layout) {
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
            ) => Self::DevicePadded,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_STANDARD,
            ) => Self::DeviceStandard,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_HOST,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
            ) => Self::HostPadded,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_HOST,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_STANDARD,
            ) => Self::HostStandard,
        }
    }
}

/// A non-owning CAGRA dataset view.
///
/// The view records the storage's residency and layout while borrowing its
/// backing tensor for `'a`. Constructing a view allocates only native metadata;
/// it never copies vector storage.
#[derive(Debug)]
pub struct DatasetView<'a> {
    handle: ffi::cuvsDatasetView_t,
    kind: DatasetKind,
    _dataset: PhantomData<&'a ()>,
}

impl<'a> DatasetView<'a> {
    /// Borrow a tensor as the host/device and padded/standard view CAGRA
    /// derives from its DLPack metadata.
    pub fn new<T>(res: &Resources, dataset: &'a T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let mut dataset_c = dataset.to_c();
        unsafe {
            let mut mem_type = std::mem::MaybeUninit::<ffi::cuvsDatasetMemType_t>::uninit();
            let mut layout = std::mem::MaybeUninit::<ffi::cuvsDatasetLayout_t>::uninit();
            check_cuvs(ffi::cuvsCagraGetDatasetMemTypeAndLayout(
                dataset_c.as_mut_ptr(),
                mem_type.as_mut_ptr(),
                layout.as_mut_ptr(),
            ))?;
            let mem_type = mem_type.assume_init();
            let layout = layout.assume_init();
            let kind = DatasetKind::from_ffi(mem_type, layout);

            let handle = init_handle(|out| {
                ffi::cuvsDatasetMakeView(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    layout,
                    mem_type,
                    out,
                )
            })?;
            Ok(Self { handle, kind, _dataset: PhantomData })
        }
    }

    /// Return this view's immutable residency/layout classification.
    pub fn kind(&self) -> DatasetKind {
        self.kind
    }

    pub(crate) fn raw(&self) -> ffi::cuvsDatasetView_t {
        self.handle
    }
}

impl Drop for DatasetView<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetViewDestroy(self.handle) }) {
            report_drop_failure("dataset view", &e);
        }
    }
}

/// Device storage owned by the caller, padded to CAGRA's required row width.
///
/// Construction performs an explicit allocation and copy. The source must be
/// device-resident; use [`DatasetView::new`] when its existing layout is
/// already suitable.
#[derive(Debug)]
pub struct DevicePaddedDataset {
    handle: ffi::cuvsDataset_t,
}

impl DevicePaddedDataset {
    /// Copy a device tensor into freshly allocated, CAGRA-padded storage.
    pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let mut dataset_c = dataset.to_c();
        let device_type = dataset_c.inner.dl_tensor.device.device_type;
        if !device_type.is_device_compatible() {
            return Err(CagraError::Validation(format!(
                "a device padded dataset requires device-resident storage, got {device_type:?}"
            )));
        }
        unsafe {
            let handle = init_handle(|out| {
                ffi::cuvsDatasetMake(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
                    ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                    out,
                )
            })?;
            Ok(Self { handle })
        }
    }

    /// Borrow this allocation as a device-padded view.
    pub fn as_view(&self) -> Result<DatasetView<'_>> {
        let handle = unsafe {
            init_handle(|out| ffi::cuvsDatasetMakeViewFromOwningPadded(self.handle, out))?
        };
        Ok(DatasetView { handle, kind: DatasetKind::DevicePadded, _dataset: PhantomData })
    }

    pub(crate) fn from_raw(handle: ffi::cuvsDataset_t) -> Self {
        Self { handle }
    }
}

impl Drop for DevicePaddedDataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetDestroy(self.handle) }) {
            report_drop_failure("device padded dataset", &e);
        }
    }
}

/// Owning standard-layout device storage returned by deserialization.
#[derive(Debug)]
pub struct StandardDataset {
    handle: ffi::cuvsDataset_t,
}

impl StandardDataset {
    pub(crate) fn from_raw(handle: ffi::cuvsDataset_t) -> Self {
        Self { handle }
    }
}

impl Drop for StandardDataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetDestroy(self.handle) }) {
            report_drop_failure("standard dataset", &e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_padded_dataset_rejects_host_storage() {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::zeros((256, 15));

        let err = DevicePaddedDataset::new(&res, &*dataset)
            .expect_err("host storage cannot back a device-padded owner");

        assert!(matches!(err, CagraError::Validation(_)), "unexpected error: {err:?}");
    }
}
