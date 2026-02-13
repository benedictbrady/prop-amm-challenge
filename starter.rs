use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};
use prop_amm_submission_sdk::{
    ffi_after_swap, ffi_compute_swap, set_return_data_bytes, set_return_data_u64,
};

const NAME: &str = "My Strategy";
const FEE_NUMERATOR: u128 = 950;
const FEE_DENOMINATOR: u128 = 1000;
const STORAGE_SIZE: usize = 1024;

#[derive(wincode::SchemaRead)]
struct ComputeSwapInstruction {
    side: u8,
    input_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
    _storage: [u8; STORAGE_SIZE],
}

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
            set_return_data_u64(output);
        }
        // tag 2 = after_swap (no-op for starter)
        2 => {
            // No storage updates needed for basic CFMM
        }
        // tag 3 = get_name (for leaderboard display)
        3 => set_return_data_bytes(NAME.as_bytes()),
        _ => {}
    }

    Ok(())
}

pub fn compute_swap(data: &[u8]) -> u64 {
    let decoded: ComputeSwapInstruction = match wincode::deserialize(data) {
        Ok(decoded) => decoded,
        Err(_) => return 0,
    };

    let side = decoded.side;
    let input_amount = decoded.input_amount as u128;
    let reserve_x = decoded.reserve_x as u128;
    let reserve_y = decoded.reserve_y as u128;

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
pub extern "C" fn compute_swap_ffi(data: *const u8, len: usize) -> u64 {
    ffi_compute_swap(data, len, compute_swap)
}

/// Optional after_swap hook for native mode.
pub fn after_swap(_data: &[u8], _storage: &mut [u8]) {
    // No-op: starter doesn't use storage
}

/// FFI export for after_swap hook (no-op for starter)
#[cfg(not(target_os = "solana"))]
#[no_mangle]
pub extern "C" fn after_swap_ffi(
    data: *const u8,
    data_len: usize,
    storage: *mut u8,
    storage_len: usize,
) {
    ffi_after_swap(data, data_len, storage, storage_len, after_swap);
}
