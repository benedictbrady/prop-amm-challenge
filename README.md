# AMM Price Function Challenge

Design a custom price function for an automated market maker. Your goal: maximize **edge** — the profit your AMM extracts from trading flow.

Your program runs inside a simulation against a benchmark AMM. Retail traders arrive, arbitrageurs keep prices efficient, and an order router splits flow between the two pools based on who offers better prices. The better your pricing, the more flow you attract and the more edge you earn.

## Quick Start

1. Copy `programs/starter/src/lib.rs` as your starting point
2. Implement your pricing logic in `compute_swap`
3. Submit your `lib.rs` source code to the web UI — the server compiles and runs it

For local development, use the CLI:

```bash
# Copy the starter template
cp programs/starter/src/lib.rs my_amm.rs

# Edit your pricing logic
edit my_amm.rs

# Run 1000 simulations locally (~5s on Apple M3 Pro)
prop-amm run my_amm.rs
```

The CLI compiles your source file and runs it natively — no toolchain setup required beyond Rust.

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

**Arbitrage**: Binary search for the optimal trade that pushes spot price to fair price. Efficient — don't try to extract value from informed flow. Trades are skipped unless expected arb profit is at least `0.01` Y (1 cent).

**Order routing**: Grid search over split ratio alpha in [0, 1]. The router picks the split that maximizes total output. Small pricing differences can shift large fractions of volume.

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

After each **real trade** (not during quoting), the engine calls your program with tag byte `2`. This lets you update your 1024-byte storage and observe the current simulation step — useful for strategies that adapt over time (dynamic fees, volatility tracking, etc.).

| Offset | Size | Field         | Type   | Description                    |
|--------|------|---------------|--------|--------------------------------|
| 0      | 1    | tag           | u8     | Always 2                       |
| 1      | 1    | side          | u8     | 0=buy X, 1=sell X              |
| 2      | 8    | input_amount  | u64    | Input token amount (1e9 scale) |
| 10     | 8    | output_amount | u64    | Output token amount (1e9 scale)|
| 18     | 8    | reserve_x     | u64    | Post-trade X reserve           |
| 26     | 8    | reserve_y     | u64    | Post-trade Y reserve           |
| 34     | 8    | step          | u64    | Current simulation step        |
| 42     | 1024 | storage       | [u8]   | Current storage (read/write)   |

To persist updated storage, call the `sol_set_storage` syscall with your modified buffer. If you don't call it, storage remains unchanged. The starter program's afterSwap is a no-op — storage is entirely optional.

**When afterSwap is called:**
- After arbitrageur executes a trade
- After router executes routed trades

**When it is NOT called:**
- During router quoting (grid search for optimal split)
- During arbitrageur quoting (binary search for optimal size)

### Requirements

| Requirement   | Description                                                        |
|---------------|--------------------------------------------------------------------|
| **Convex**    | Marginal price must worsen with trade size. Non-convex programs are rejected. |
| **Monotonic** | Larger input must produce larger output.                           |
| **< 100k CU** | Must execute within the compute unit limit.                       |

## Writing a Program

Start with `programs/starter/` — a constant-product AMM with 500 bps fees. The key pieces:

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

## Local Development (CLI)

The CLI compiles and runs your `.rs` source file directly — no manual build step needed.

```bash
# Run simulations (default: 1000 sims, 10k steps each)
prop-amm run my_amm.rs

# Fewer sims for quick iteration
prop-amm run my_amm.rs --simulations 10

# Build only (native + BPF artifacts)
prop-amm build my_amm.rs

# Validate convexity and monotonicity
prop-amm validate my_amm.rs
```

The 30 bps normalizer typically scores around 250-350 edge per simulation.

### Native vs BPF

By default, `prop-amm run` compiles your program as a **native shared library** and runs it directly. This is fast enough for rapid iteration — 1,000 simulations complete in seconds.

BPF mode (`--bpf`) runs your program through the Solana BPF interpreter, which is **~100x slower**. Use it only as a final check before submitting to verify your program compiles and behaves correctly under the BPF runtime. Don't use it for day-to-day development.

```bash
# Fast iteration (native, default)
prop-amm run my_amm.rs

# Final validation before submission (BPF, slow)
prop-amm run my_amm.rs --bpf --simulations 10
```

The engine parallelizes across simulations using up to 8 worker threads (configurable with `--workers`).

| Workload                  | Time           | Platform         |
|---------------------------|----------------|------------------|
| 1,000 sims / 10k steps   | ~5s            | Apple M3 Pro, native |
| 1,000 sims / 10k steps   | ~15 min        | Apple M3 Pro, BPF |

## Submission

Submit your `lib.rs` source code through the web UI. The server handles compilation, validation, and simulation — you don't need any toolchain beyond what's needed for local testing.

The server validates your program (monotonicity, convexity), then runs 1,000 simulations against the normalizer. Local results may diverge slightly from submission scores due to different RNG seeds and hyperparameter variance.

### Restrictions

Your submitted source code must be a single `lib.rs` file. The only allowed dependency is `pinocchio` (for Solana BPF syscalls). The following are blocked for security:

- `include!()`, `include_str!()`, `include_bytes!()` (compile-time file access)
- `env!()`, `option_env!()` (compile-time environment access)
- `extern crate` declarations
- External module files (`mod foo;`)
