use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};
use prop_amm_submission_sdk::{set_return_data_bytes, set_return_data_u64, set_storage};

const NAME: &str = "Adaptive Edge V1";
const MODEL_USED: &str = "Claude Opus 4.6"; 
const STORAGE_SIZE: usize = 1024;

// Fee parameters (in basis points)
const BASE_FEE_BPS: u128 = 45;    // Slightly below normalizer average (55 bps)
const MIN_FEE_BPS: u128 = 25;     // Floor: still profitable
const MAX_FEE_BPS: u128 = 120;    // Ceiling: don't lose all flow

// Storage layout:
// [0..8]   buy_volume: u64
// [8..16]  sell_volume: u64
// [16..24] arb_indicator: u64 (large one-sided trades = arb)
// [24..32] trade_count: u64
// [32..40] current_fee_bps: u64

#[derive(wincode::SchemaRead)]
struct ComputeSwapInstruction {
    side: u8,
    input_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
    storage: [u8; STORAGE_SIZE],
}

#[derive(wincode::SchemaRead)]
struct AfterSwapInstruction {
    tag: u8,
    side: u8,
    input_amount: u64,
    output_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
    step: u64,
    storage: [u8; STORAGE_SIZE],
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
        0 | 1 => {
            let output = compute_swap(instruction_data);
            set_return_data_u64(output);
        }
        2 => {
            after_swap_handler(instruction_data);
        }
        3 => set_return_data_bytes(NAME.as_bytes()),
        4 => set_return_data_bytes(get_model_used().as_bytes()),
        _ => {}
    }

    Ok(())
}

pub fn get_model_used() -> &'static str {
    MODEL_USED
}

// Helper to read u64 from storage
fn read_u64_le(storage: &[u8], offset: usize) -> u64 {
    if offset + 8 > storage.len() {
        return 0;
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&storage[offset..offset + 8]);
    u64::from_le_bytes(bytes)
}

// Helper to write u64 to storage
fn write_u64_le(storage: &mut [u8], offset: usize, value: u64) {
    if offset + 8 <= storage.len() {
        storage[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

// Get current fee from storage, or default
fn get_current_fee_bps(storage: &[u8]) -> u128 {
    let stored = read_u64_le(storage, 32);
    if stored == 0 {
        BASE_FEE_BPS
    } else {
        (stored as u128).clamp(MIN_FEE_BPS, MAX_FEE_BPS)
    }
}

pub fn compute_swap(data: &[u8]) -> u64 {
    let decoded: ComputeSwapInstruction = match wincode::deserialize(data) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let input = decoded.input_amount as u128;
    let rx = decoded.reserve_x as u128;
    let ry = decoded.reserve_y as u128;

    if rx == 0 || ry == 0 || input == 0 {
        return 0;
    }

    // Dynamic fee from storage
    let fee_bps = get_current_fee_bps(&decoded.storage);
    let fee_num = 10000u128 - fee_bps;  // e.g., 45 bps → 9955/10000
    let fee_den = 10000u128;

    let k = rx * ry;

    match decoded.side {
        0 => {
            // Buy X: Y in, X out
            let net_input = input * fee_num / fee_den;
            let new_ry = ry + net_input;
            let new_rx = (k + new_ry - 1) / new_ry;  // round up
            rx.saturating_sub(new_rx) as u64
        }
        1 => {
            // Sell X: X in, Y out
            let net_input = input * fee_num / fee_den;
            let new_rx = rx + net_input;
            let new_ry = (k + new_rx - 1) / new_rx;  // round up
            ry.saturating_sub(new_ry) as u64
        }
        _ => 0,
    }
}

fn after_swap_handler(data: &[u8]) {
    let decoded: AfterSwapInstruction = match wincode::deserialize(data) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut storage = decoded.storage;
    let input = decoded.input_amount as u128;
    let rx = decoded.reserve_x as u128;
    let ry = decoded.reserve_y as u128;

    // Track volume by side
    if decoded.side == 0 {
        let vol = read_u64_le(&storage, 0);
        write_u64_le(&mut storage, 0, vol.saturating_add(decoded.input_amount));
    } else {
        let vol = read_u64_le(&storage, 8);
        write_u64_le(&mut storage, 8, vol.saturating_add(decoded.input_amount));
    }

    // Update trade count
    let count = read_u64_le(&storage, 24) + 1;
    write_u64_le(&mut storage, 24, count);

    // Detect arb activity: large trades relative to reserves suggest arbitrage
    // Arb trades push price toward fair value, usually large and one-sided
    let trade_size_ratio = if decoded.side == 0 {
        (input * 10000) / ry  // % of Y reserve
    } else {
        (input * 10000) / rx  // % of X reserve
    };

    // EMA of trade size ratio (proxy for arb intensity)
    let old_arb = read_u64_le(&storage, 16) as u128;
    let alpha = 50u128;  // 5% weight for new observation
    let new_arb = (old_arb * (1000 - alpha) + trade_size_ratio * alpha) / 1000;
    write_u64_le(&mut storage, 16, new_arb as u64);

    // Adaptive fee logic:
    // - High arb indicator → increase fee (protect from informed flow)
    // - Low arb indicator → decrease fee (capture more retail)
    // 
    // arb_indicator ~100 = 1% average trade size = normal retail
    // arb_indicator ~500 = 5% average trade size = heavy arb
    let arb_adjustment = if new_arb > 300 {
        // High arb: increase fee
        ((new_arb - 300) * 30 / 200).min(75)  // up to +75 bps
    } else if new_arb < 100 {
        // Low arb: decrease fee to capture more flow
        ((100 - new_arb) * 20 / 100).min(20)  // up to -20 bps
    } else {
        0
    };

    let current_fee = get_current_fee_bps(&storage);
    let new_fee = if new_arb > 300 {
        current_fee + arb_adjustment
    } else {
        current_fee.saturating_sub(arb_adjustment)
    };

    // Smooth fee changes (don't jump too fast)
    let smoothed_fee = (current_fee * 9 + new_fee) / 10;
    write_u64_le(&mut storage, 32, smoothed_fee.clamp(MIN_FEE_BPS, MAX_FEE_BPS) as u64);

    set_storage(&storage);
}

/// Native hook for local testing
pub fn after_swap(_data: &[u8], _storage: &mut [u8]) {
    // Simplified for native - real logic in after_swap_handler
}