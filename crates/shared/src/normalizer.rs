/// Native normalizer swap function (30bp CFMM).
/// Takes instruction data (25+ bytes, extra storage bytes ignored), returns output amount.
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

    let fee_bps = if data.len() >= 27 {
        let raw = u16::from_le_bytes([data[25], data[26]]);
        if raw == 0 { 30u128 } else { raw as u128 }
    } else {
        30u128
    };

    let k = reserve_x * reserve_y;

    match side {
        0 => {
            let net = input_amount * (10000 - fee_bps) / 10000;
            let new_ry = reserve_y + net;
            reserve_x.saturating_sub((k + new_ry - 1) / new_ry) as u64
        }
        1 => {
            let net = input_amount * (10000 - fee_bps) / 10000;
            let new_rx = reserve_x + net;
            reserve_y.saturating_sub((k + new_rx - 1) / new_rx) as u64
        }
        _ => 0,
    }
}

/// Native normalizer after_swap hook (no-op).
pub fn after_swap(_data: &[u8], _storage: &mut [u8]) {
    // No-op: normalizer doesn't use storage
}
