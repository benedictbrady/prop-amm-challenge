# AMM Price Function Challenge

Design a custom price function for an automated market maker. Your goal: maximize **edge** — the profit your AMM extracts from trading flow.

Your program runs inside a simulation against a benchmark AMM. Retail traders arrive, arbitrageurs keep prices efficient, and an order router splits flow between the two pools based on who offers better prices. The better your pricing, the more flow you attract and the more edge you earn.

## Quick Start

```bash
# Copy the starter template
cp -r programs/starter programs/my-strategy
# Edit programs/my-strategy/Cargo.toml — change the package name
# Edit programs/my-strategy/src/lib.rs — implement your pricing logic

# Build native + BPF
prop-amm build programs/my-strategy

# Run 1000 simulations locally (~5s on Apple M3 Pro)
prop-amm run programs/my-strategy/target/release/libmy_strategy.dylib

# Validate BPF program before submitting
prop-amm validate programs/my-strategy/target/deploy/my_strategy.so

# Submit the .so file to the API
```

## How the Simulation Works

Each simulation runs **10,000 steps**. At each step:

1. **Fair price moves** via geometric Brownian motion
2. **Arbitrageurs trade** — they push each AMM's spot price toward the fair price, extracting profit from stale quotes
3. **Retail orders arrive** — random buy/sell orders, routed optimally across both AMMs

Your program competes against a **normalizer AMM** — a constant-product market maker with 30 bps fees. Both start with identical reserves (100 X, 10,000 Y at price 100).

### Why the Normalizer Matters

Without competition, setting 10% fees would appear profitable — huge spreads on the few trades that execute. The normalizer prevents this: if your pricing is too aggressive, retail routes to the 30 bps pool and you get nothing.

There's no free lunch from slightly undercutting either. The optimal strategy depends on market conditions, trade patterns, and how you manage the tradeoff between spread revenue and adverse selection.

### Market Parameters

**Price process**: `S(t+1) = S(t) * exp(-sigma^2/2 + sigma*Z)` where `Z ~ N(0,1)`
- No drift (mu = 0)
- Per-step volatility varies across simulations: `sigma ~ U[0.088%, 0.101%]`

**Retail flow**: Poisson arrival, log-normal sizes, 50/50 buy/sell
- Arrival rate `lambda ~ U[0.6, 1.0]` per step
- Mean order size `~ U[19, 21]` in Y terms

**Arbitrage**: Deterministic probe search over retail-like candidate sizes (with fixed tail exploration) to find profitable clips and push spot price toward fair price. Trades are skipped unless expected arb profit is above `min_arb_profit` (default `0.01` Y, i.e. 1 cent).

**Order routing**: Deterministic probe search over split size to maximize total output across both pools. Small pricing differences can shift large fractions of volume.

### Edge

Edge measures profitability using the fair price at trade time:

```
For each trade on your AMM:
  Sell X (AMM receives X, pays Y):  edge = amount_x * fair_price - amount_y
  Buy X  (AMM receives Y, pays X):  edge = amount_y - amount_x * fair_price
```

Retail trades produce positive edge (you profit from the spread). Arbitrage trades produce negative edge (you lose to informed flow). Good strategies maximize the former while minimizing the latter.

## Program Interface

### compute_swap

Your program receives instruction data with reserves and a 1024-byte read-only storage buffer:

| Offset | Size | Field        | Type   | Description                    |
|--------|------|--------------|--------|--------------------------------|
| 0      | 1    | side         | u8     | 0=buy X (Y input), 1=sell X   |
| 1      | 8    | input_amount | u64    | Input token amount (1e9 scale) |
| 9      | 8    | reserve_x    | u64    | Current X reserve (1e9 scale)  |
| 17     | 8    | reserve_y    | u64    | Current Y reserve (1e9 scale)  |
| 25     | 1024 | storage      | [u8]   | Read-only strategy storage     |

Return 8 bytes via `sol_set_return_data` — the `output_amount: u64` (1e9 scale).

### afterSwap (Optional)

After each **real trade** (not during quoting), the engine calls your program with tag byte `2`. This lets you update your 1024-byte storage — useful for strategies that adapt over time (dynamic fees, volatility tracking, etc.).

| Offset | Size | Field         | Type   | Description                    |
|--------|------|---------------|--------|--------------------------------|
| 0      | 1    | tag           | u8     | Always 2                       |
| 1      | 1    | side          | u8     | 0=buy X, 1=sell X              |
| 2      | 8    | input_amount  | u64    | Input token amount (1e9 scale) |
| 10     | 8    | output_amount | u64    | Output token amount (1e9 scale)|
| 18     | 8    | reserve_x     | u64    | Post-trade X reserve           |
| 26     | 8    | reserve_y     | u64    | Post-trade Y reserve           |
| 34     | 1024 | storage       | [u8]   | Current storage (read/write)   |

To persist updated storage, call the `sol_set_storage` syscall with your modified buffer. If you don't call it, storage remains unchanged. The starter program's afterSwap is a no-op — storage is entirely optional.

**When afterSwap is called:**
- After arbitrageur executes a trade
- After router executes routed trades

**When it is NOT called:**
- During router quoting (deterministic probe search for optimal split)
- During arbitrageur quoting (deterministic probe search for profitable size)

### Requirements

| Requirement   | Description                                                        |
|---------------|--------------------------------------------------------------------|
| **Convex**    | Marginal price must worsen with trade size. Non-convex programs are rejected. |
| **Monotonic** | Larger input must produce larger output.                           |
| **< 100k CU** | Must execute within the compute unit limit.                       |

## Writing a Program

Start with `programs/starter/` — a constant-product AMM with 50 bps fees. The key pieces:

```rust
use pinocchio::{account_info::AccountInfo, entrypoint, pubkey::Pubkey, ProgramResult};

#[cfg(not(feature = "no-entrypoint"))]
entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey, _accounts: &[AccountInfo], instruction_data: &[u8],
) -> ProgramResult {
    match instruction_data[0] {
        0 | 1 => {  // compute_swap
            let output = compute_swap(instruction_data);
            unsafe { pinocchio::syscalls::sol_set_return_data(output.to_le_bytes().as_ptr(), 8); }
        }
        2 => {      // afterSwap — update storage here if needed
        }
        _ => {}
    }
    Ok(())
}

pub fn compute_swap(data: &[u8]) -> u64 {
    let side = data[0];
    let input = u64::from_le_bytes(data[1..9].try_into().unwrap()) as u128;
    let rx = u64::from_le_bytes(data[9..17].try_into().unwrap()) as u128;
    let ry = u64::from_le_bytes(data[17..25].try_into().unwrap()) as u128;
    // Storage available at data[25..1049] if needed

    // Your pricing logic here...
    todo!()
}

/// Required for native execution (local testing).
#[cfg(not(target_os = "solana"))]
#[no_mangle]
pub unsafe extern "C" fn compute_swap_ffi(data: *const u8, len: usize) -> u64 {
    compute_swap(core::slice::from_raw_parts(data, len))
}

/// Optional: afterSwap hook for native mode.
#[cfg(not(target_os = "solana"))]
#[no_mangle]
pub unsafe extern "C" fn after_swap_ffi(
    _data: *const u8, _data_len: usize, _storage: *mut u8, _storage_len: usize,
) {
    // Update storage here if needed
}
```

### Tips

- Use `u128` intermediates to avoid overflow (reserves at 1e9 scale can multiply to ~1e24)
- Test convexity with `prop-amm validate` before running simulations
- Think about how your marginal price schedule affects the routing split
- The arbitrageur is efficient — don't try to extract value from informed flow
- Storage is zero-initialized at the start of each simulation and persists across all trades within a simulation

## CLI

```bash
# Build native + BPF
prop-amm build programs/my-strategy

# Run simulations (default: 1000 sims, 10k steps each)
prop-amm run <path-to-native-lib>

# Fewer sims for quick iteration
prop-amm run <path-to-native-lib> --simulations 10

# Validate BPF program
prop-amm validate <path-to-bpf.so>
```

The 30 bps normalizer typically scores around 250-350 edge per simulation.

## Performance

Simulations run natively via `dlopen` — your Rust code executes directly, no BPF interpreter overhead. BPF is only used for the submission `.so` file validation.

| Workload                  | Time           | Platform         |
|---------------------------|----------------|------------------|
| 1,000 sims / 10k steps   | ~5s            | Apple M3 Pro, 12 cores |
| Single sim / 10k steps   | ~5ms           | Native execution |

The engine parallelizes across simulations using up to 8 worker threads by default.

## Submission

Build your BPF `.so` and upload it to the submission API:

```bash
cargo build-sbf --manifest-path programs/my-strategy/Cargo.toml
# Upload: programs/my-strategy/target/deploy/my_strategy.so
```

Local results may diverge slightly from submission scores due to different RNG seeds and hyperparameter variance. Run more simulations locally to reduce variance.
