/// Instruction data layout for compute_swap (25 bytes base + 1024 storage):
/// | Offset    | Size | Field        | Type | Description                    |
/// |-----------|------|--------------|------|--------------------------------|
/// | 0         | 1    | side         | u8   | 0=buy X (Y input), 1=sell X   |
/// | 1         | 8    | input_amount | u64  | Input token amount (1e9 scale) |
/// | 9         | 8    | reserve_x    | u64  | Current X reserve (1e9 scale)  |
/// | 17        | 8    | reserve_y    | u64  | Current Y reserve (1e9 scale)  |
/// | 25        | 1024 | storage      | [u8] | Read-only strategy storage     |

pub const INSTRUCTION_SIZE: usize = 25;
pub const STORAGE_SIZE: usize = 1024;
pub const SWAP_INSTRUCTION_SIZE: usize = INSTRUCTION_SIZE + STORAGE_SIZE; // 1049

/// after_swap instruction layout (1058 bytes):
/// | Offset    | Size | Field         | Type | Description                    |
/// |-----------|------|---------------|------|--------------------------------|
/// | 0         | 1    | tag           | u8   | Always 2                       |
/// | 1         | 1    | side          | u8   | 0=buy X, 1=sell X              |
/// | 2         | 8    | input_amount  | u64  | Input token amount (1e9 scale) |
/// | 10        | 8    | output_amount | u64  | Output token amount            |
/// | 18        | 8    | reserve_x     | u64  | Post-trade X reserve           |
/// | 26        | 8    | reserve_y     | u64  | Post-trade Y reserve           |
/// | 34        | 1024 | storage       | [u8] | Current storage state          |
pub const AFTER_SWAP_SIZE: usize = 34 + STORAGE_SIZE; // 1058

pub fn encode_instruction(
    side: u8,
    input_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
) -> [u8; INSTRUCTION_SIZE] {
    let mut data = [0u8; INSTRUCTION_SIZE];
    data[0] = side;
    data[1..9].copy_from_slice(&input_amount.to_le_bytes());
    data[9..17].copy_from_slice(&reserve_x.to_le_bytes());
    data[17..25].copy_from_slice(&reserve_y.to_le_bytes());
    data
}

pub fn decode_instruction(data: &[u8]) -> (u8, u64, u64, u64) {
    let side = data[0];
    let input_amount = u64::from_le_bytes(data[1..9].try_into().unwrap());
    let reserve_x = u64::from_le_bytes(data[9..17].try_into().unwrap());
    let reserve_y = u64::from_le_bytes(data[17..25].try_into().unwrap());
    (side, input_amount, reserve_x, reserve_y)
}

pub fn encode_swap_instruction(
    side: u8,
    input_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
    storage: &[u8],
) -> Vec<u8> {
    let mut data = vec![0u8; SWAP_INSTRUCTION_SIZE];
    data[0] = side;
    data[1..9].copy_from_slice(&input_amount.to_le_bytes());
    data[9..17].copy_from_slice(&reserve_x.to_le_bytes());
    data[17..25].copy_from_slice(&reserve_y.to_le_bytes());
    let copy_len = storage.len().min(STORAGE_SIZE);
    data[25..25 + copy_len].copy_from_slice(&storage[..copy_len]);
    data
}

pub fn encode_after_swap(
    side: u8,
    input_amount: u64,
    output_amount: u64,
    reserve_x: u64,
    reserve_y: u64,
    storage: &[u8],
) -> Vec<u8> {
    let mut data = vec![0u8; AFTER_SWAP_SIZE];
    data[0] = 2; // tag
    data[1] = side;
    data[2..10].copy_from_slice(&input_amount.to_le_bytes());
    data[10..18].copy_from_slice(&output_amount.to_le_bytes());
    data[18..26].copy_from_slice(&reserve_x.to_le_bytes());
    data[26..34].copy_from_slice(&reserve_y.to_le_bytes());
    let copy_len = storage.len().min(STORAGE_SIZE);
    data[34..34 + copy_len].copy_from_slice(&storage[..copy_len]);
    data
}

pub fn decode_after_swap(data: &[u8]) -> (u8, u64, u64, u64, u64, &[u8]) {
    let side = data[1];
    let input_amount = u64::from_le_bytes(data[2..10].try_into().unwrap());
    let output_amount = u64::from_le_bytes(data[10..18].try_into().unwrap());
    let reserve_x = u64::from_le_bytes(data[18..26].try_into().unwrap());
    let reserve_y = u64::from_le_bytes(data[26..34].try_into().unwrap());
    let storage = &data[34..];
    (
        side,
        input_amount,
        output_amount,
        reserve_x,
        reserve_y,
        storage,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let side = 1u8;
        let amount = 123_456_789_000u64;
        let rx = 100_000_000_000u64;
        let ry = 10_000_000_000_000u64;

        let encoded = encode_instruction(side, amount, rx, ry);
        let (s, a, x, y) = decode_instruction(&encoded);

        assert_eq!(s, side);
        assert_eq!(a, amount);
        assert_eq!(x, rx);
        assert_eq!(y, ry);
    }

    #[test]
    fn test_swap_instruction_with_storage() {
        let storage = [0xAB; STORAGE_SIZE];
        let data = encode_swap_instruction(0, 1000, 2000, 3000, &storage);
        assert_eq!(data.len(), SWAP_INSTRUCTION_SIZE);
        let (side, amount, rx, ry) = decode_instruction(&data);
        assert_eq!(side, 0);
        assert_eq!(amount, 1000);
        assert_eq!(rx, 2000);
        assert_eq!(ry, 3000);
        assert_eq!(&data[25..], &storage[..]);
    }

    #[test]
    fn test_after_swap_roundtrip() {
        let storage = [0xCD; STORAGE_SIZE];
        let data = encode_after_swap(1, 100, 200, 300, 400, &storage);
        assert_eq!(data.len(), AFTER_SWAP_SIZE);
        let (side, inp, out, rx, ry, stor) = decode_after_swap(&data);
        assert_eq!(side, 1);
        assert_eq!(inp, 100);
        assert_eq!(out, 200);
        assert_eq!(rx, 300);
        assert_eq!(ry, 400);
        assert_eq!(stor, &storage[..]);
    }
}
