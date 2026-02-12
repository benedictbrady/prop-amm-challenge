use prop_amm_shared::instruction::{encode_after_swap, encode_swap_instruction, STORAGE_SIZE};

/// A swap function signature: takes instruction data (with storage appended), returns output amount.
pub type SwapFn = fn(&[u8]) -> u64;

/// An after_swap function signature: takes (trade_info, mutable_storage).
pub type AfterSwapFn = fn(&[u8], &mut [u8]);

/// Native executor that calls a Rust function directly (no BPF overhead).
#[derive(Clone)]
pub struct NativeExecutor {
    swap_fn: SwapFn,
    after_swap_fn: Option<AfterSwapFn>,
}

impl NativeExecutor {
    pub fn new(swap_fn: SwapFn, after_swap_fn: Option<AfterSwapFn>) -> Self {
        Self {
            swap_fn,
            after_swap_fn,
        }
    }

    #[inline]
    pub fn execute(&self, side: u8, amount: u64, rx: u64, ry: u64, storage: &[u8]) -> u64 {
        let data = encode_swap_instruction(side, amount, rx, ry, storage);
        (self.swap_fn)(&data)
    }

    #[inline]
    pub fn execute_after_swap(
        &self,
        side: u8,
        input_amount: u64,
        output_amount: u64,
        rx: u64,
        ry: u64,
        step: u64,
        storage: &mut [u8],
    ) {
        if let Some(after_swap) = self.after_swap_fn {
            let data = encode_after_swap(side, input_amount, output_amount, rx, ry, step, storage);
            let copy_len = storage.len().min(STORAGE_SIZE);
            after_swap(&data, &mut storage[..copy_len]);
        }
    }
}
