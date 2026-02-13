#![cfg_attr(target_os = "solana", no_std)]

pub const STORAGE_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageError {
    TooLarge,
}

#[inline]
pub fn set_return_data_u64(value: u64) {
    set_return_data_bytes(&value.to_le_bytes());
}

#[inline]
pub fn set_return_data_bytes(bytes: &[u8]) {
    pinocchio::program::set_return_data(bytes);
}

#[inline]
pub fn set_storage(storage: &[u8]) -> Result<(), StorageError> {
    if storage.len() > STORAGE_SIZE {
        return Err(StorageError::TooLarge);
    }
    #[cfg(target_os = "solana")]
    unsafe {
        sol_set_storage(storage.as_ptr(), storage.len() as u64);
    }
    Ok(())
}

#[cfg(target_os = "solana")]
extern "C" {
    fn sol_set_storage(data: *const u8, length: u64);
}

/// Safe wrapper for native entrypoint glue.
///
/// This keeps unsafe pointer handling in the SDK so user submissions can stay
/// fully safe Rust.
#[cfg(not(target_os = "solana"))]
#[inline]
pub fn ffi_compute_swap(data: *const u8, len: usize, compute_swap: fn(&[u8]) -> u64) -> u64 {
    if data.is_null() {
        return if len == 0 { compute_swap(&[]) } else { 0 };
    }
    let slice = unsafe { core::slice::from_raw_parts(data, len) };
    compute_swap(slice)
}

/// Safe wrapper for native after_swap glue.
///
/// Null pointers are treated as invalid when their corresponding length is
/// non-zero and become no-ops.
#[cfg(not(target_os = "solana"))]
#[inline]
pub fn ffi_after_swap(
    data: *const u8,
    data_len: usize,
    storage: *mut u8,
    storage_len: usize,
    after_swap: fn(&[u8], &mut [u8]),
) {
    if (data.is_null() && data_len != 0) || (storage.is_null() && storage_len != 0) {
        return;
    }

    let data_slice = if data_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(data, data_len) }
    };

    let storage_slice = if storage_len == 0 {
        &mut []
    } else {
        unsafe { core::slice::from_raw_parts_mut(storage, storage_len) }
    };

    after_swap(data_slice, storage_slice);
}
