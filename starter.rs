use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};

const NAME: &str = "My Strategy";
const FEE_NUMERATOR: u128 = 950;
const FEE_DENOMINATOR: u128 = 1000;

#[cfg(not(feature = "no-entrypoint"))]
entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    if instruction_data.is_empty() {
        return Ok(());
    }

    match instruction_data[0] {
        // tag 0 or 1 = compute_swap (side)
        0 | 1 => {
            let output = compute_swap(instruction_data);
            unsafe {
                pinocchio::syscalls::sol_set_return_data(output.to_le_bytes().as_ptr(), 8);
            }
        }
        // tag 2 = after_swap (no-op for starter)
        2 => {
            // No storage updates needed for basic CFMM
        }
        _ => {}
    }

    Ok(())
}

pub fn compute_swap(data: &[u8]) -> u64 {
    if data.len() < 25 {
        return 0;
    }

    let side = data[0];
    let input_amount = u64::from_le_bytes([
        data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
    ]) as u128;
    let reserve_x = u64::from_le_bytes([
        data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16],
    ]) as u128;
    let reserve_y = u64::from_le_bytes([
        data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24],
    ]) as u128;

    if reserve_x == 0 || reserve_y == 0 {
        return 0;
    }

    let k = reserve_x * reserve_y;

    match side {
        0 => {
            let net_y = input_amount * FEE_NUMERATOR / FEE_DENOMINATOR;
            let new_ry = reserve_y + net_y;
            let k_div = (k + new_ry - 1) / new_ry;
            reserve_x.saturating_sub(k_div) as u64
        }
        1 => {
            let net_x = input_amount * FEE_NUMERATOR / FEE_DENOMINATOR;
            let new_rx = reserve_x + net_x;
            let k_div = (k + new_rx - 1) / new_rx;
            reserve_y.saturating_sub(k_div) as u64
        }
        _ => 0,
    }
}

/// FFI export for native shared library loading (used by prop-amm run)
#[cfg(not(target_os = "solana"))]
#[no_mangle]
pub unsafe extern "C" fn compute_swap_ffi(data: *const u8, len: usize) -> u64 {
    compute_swap(core::slice::from_raw_parts(data, len))
}

/// FFI export for after_swap hook (no-op for starter)
#[cfg(not(target_os = "solana"))]
#[no_mangle]
pub unsafe extern "C" fn after_swap_ffi(
    _data: *const u8,
    _data_len: usize,
    _storage: *mut u8,
    _storage_len: usize,
) {
    // No-op: starter doesn't use storage
}
