use prop_amm_executor::{BpfExecutor, BpfProgram};
use prop_amm_shared::instruction::STORAGE_SIZE;
use prop_amm_shared::nano::NANO_SCALE_F64;
use prop_amm_shared::nano::{f64_to_nano, nano_to_f64};

use super::compile;

pub fn run(file: &str) -> anyhow::Result<()> {
    println!("Compiling {} (BPF)...", file);
    let so_path = compile::compile_bpf(file)?;

    println!("Validating program: {}", so_path.display());

    // Load ELF
    let elf_bytes = std::fs::read(&so_path)?;
    let program =
        BpfProgram::load(&elf_bytes).map_err(|e| anyhow::anyhow!("Failed to load ELF: {}", e))?;
    println!("  [PASS] ELF loaded and verified");

    let mut executor = BpfExecutor::new(program);
    let storage = [0u8; STORAGE_SIZE];

    // Basic execution test
    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let buy_output = executor
        .execute(0, f64_to_nano(10.0), rx, ry, &storage)
        .map_err(|e| anyhow::anyhow!("Buy execution failed: {}", e))?;
    if buy_output == 0 {
        anyhow::bail!("FAIL: Buy X returned zero output");
    }
    println!(
        "  [PASS] Buy X: input_y=10.0 -> output_x={:.6}",
        nano_to_f64(buy_output)
    );

    let sell_output = executor
        .execute(1, f64_to_nano(1.0), rx, ry, &storage)
        .map_err(|e| anyhow::anyhow!("Sell execution failed: {}", e))?;
    if sell_output == 0 {
        anyhow::bail!("FAIL: Sell X returned zero output");
    }
    println!(
        "  [PASS] Sell X: input_x=1.0 -> output_y={:.6}",
        nano_to_f64(sell_output)
    );

    // Monotonicity check: larger input -> larger output
    println!("  Checking monotonicity...");
    let trade_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];

    // Buy side monotonicity
    let mut prev_output = 0u64;
    for &size in &trade_sizes {
        let output = executor
            .execute(0, f64_to_nano(size), rx, ry, &storage)
            .map_err(|e| anyhow::anyhow!("Execution failed at size {}: {}", size, e))?;
        if output <= prev_output && prev_output > 0 {
            anyhow::bail!(
                "FAIL: Monotonicity violation (buy side). size={} output={} <= prev_output={}",
                size,
                output,
                prev_output
            );
        }
        prev_output = output;
    }
    println!("  [PASS] Buy side monotonicity");

    // Sell side monotonicity
    prev_output = 0;
    for &size in &trade_sizes {
        let output = executor
            .execute(1, f64_to_nano(size), rx, ry, &storage)
            .map_err(|e| anyhow::anyhow!("Execution failed at size {}: {}", size, e))?;
        if output <= prev_output && prev_output > 0 {
            anyhow::bail!(
                "FAIL: Monotonicity violation (sell side). size={} output={} <= prev_output={}",
                size,
                output,
                prev_output
            );
        }
        prev_output = output;
    }
    println!("  [PASS] Sell side monotonicity");

    // Convexity check: marginal price must worsen with size
    println!("  Checking convexity...");
    let eps = 0.001;

    // Buy side convexity: marginal X output per Y input should decrease
    let mut prev_marginal = f64::MAX;
    for &size in &trade_sizes {
        let out_lo = nano_to_f64(executor.execute(0, f64_to_nano(size), rx, ry, &storage)?);
        let out_hi = nano_to_f64(executor.execute(0, f64_to_nano(size + eps), rx, ry, &storage)?);
        let marginal = (out_hi - out_lo) / eps;

        if marginal > prev_marginal + 1e-9 {
            anyhow::bail!(
                "FAIL: Convexity violation (buy side). At size={}, marginal={:.9} > prev={:.9}",
                size,
                marginal,
                prev_marginal
            );
        }
        prev_marginal = marginal;
    }
    println!("  [PASS] Buy side convexity");

    // Sell side convexity: marginal Y output per X input should decrease
    prev_marginal = f64::MAX;
    for &size in &trade_sizes {
        let out_lo = nano_to_f64(executor.execute(1, f64_to_nano(size), rx, ry, &storage)?);
        let out_hi = nano_to_f64(executor.execute(1, f64_to_nano(size + eps), rx, ry, &storage)?);
        let marginal = (out_hi - out_lo) / eps;

        if marginal > prev_marginal + 1e-9 {
            anyhow::bail!(
                "FAIL: Convexity violation (sell side). At size={}, marginal={:.9} > prev={:.9}",
                size,
                marginal,
                prev_marginal
            );
        }
        prev_marginal = marginal;
    }
    println!("  [PASS] Sell side convexity");

    // Randomized behavioral checks over varied reserve/storage states
    println!("  Checking randomized reserve/storage states...");
    for seed in 0..32u64 {
        let mut storage = [0u8; STORAGE_SIZE];
        for i in 0..32usize {
            storage[i] = (mix(seed.wrapping_add(i as u64)) & 0xFF) as u8;
        }

        let rx = 1_000_000_000u64 + (mix(seed ^ 0x0123_4567_89AB_CDEF) % 2_000_000_000_000u64);
        let ry = 1_000_000_000u64 + (mix(seed ^ 0x0F0F_0F0F_F0F0_F0F0) % 200_000_000_000_000u64);

        for side in [0u8, 1u8] {
            check_curve_shape(&mut executor, side, rx, ry, &storage).map_err(|e| {
                anyhow::anyhow!(
                    "Randomized curve check failed (seed={}, side={}): {}",
                    seed,
                    side,
                    e
                )
            })?;
        }

        // Exercise after_swap and then re-check quote behavior with updated storage.
        let side = (seed & 1) as u8;
        let amount = 1_000_000 + (mix(seed ^ 0xDEAD_BEEF) % 10_000_000_000);
        let out = executor.execute(side, amount, rx, ry, &storage)?;
        let (post_rx, post_ry) = if side == 0 {
            (rx.saturating_sub(out), ry.saturating_add(amount))
        } else {
            (rx.saturating_add(amount), ry.saturating_sub(out))
        };
        executor.execute_after_swap(side, amount, out, post_rx, post_ry, seed, &mut storage)?;

        for side in [0u8, 1u8] {
            check_curve_shape(
                &mut executor,
                side,
                post_rx.max(1),
                post_ry.max(1),
                &storage,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Post-after_swap curve check failed (seed={}, side={}): {}",
                    seed,
                    side,
                    e
                )
            })?;
        }
    }
    println!("  [PASS] Randomized reserve/storage checks");

    println!("\nAll validation checks passed!");
    Ok(())
}

fn check_curve_shape(
    executor: &mut BpfExecutor,
    side: u8,
    rx: u64,
    ry: u64,
    storage: &[u8; STORAGE_SIZE],
) -> anyhow::Result<()> {
    let reserve_out = if side == 0 { rx } else { ry };
    let reserve_in = if side == 0 { ry } else { rx };
    if reserve_out <= 1 || reserve_in <= 1 {
        anyhow::bail!("insufficient reserves");
    }

    let max_in = (reserve_in / 5).max(1_000_000);
    let mut inputs = Vec::with_capacity(10);
    for i in 1..=10u64 {
        let amount = ((max_in as u128 * i as u128) / 10u128) as u64;
        let amount = amount.max(1_000_000);
        if inputs.last().copied() != Some(amount) {
            inputs.push(amount);
        }
    }
    if inputs.len() < 2 {
        anyhow::bail!("not enough unique test inputs");
    }

    let mut outputs = Vec::with_capacity(inputs.len());
    let mut prev_out = 0u64;
    for &input in &inputs {
        let out = executor.execute(side, input, rx, ry, storage)?;
        if out == 0 {
            anyhow::bail!("zero output for input {}", input);
        }
        if out > reserve_out {
            anyhow::bail!("output {} exceeds reserve {}", out, reserve_out);
        }
        if out <= prev_out {
            anyhow::bail!(
                "non-monotonic output at input {}: {} <= {}",
                input,
                out,
                prev_out
            );
        }
        prev_out = out;
        outputs.push(out);
    }

    let mut prev_slope = f64::INFINITY;
    for i in 0..(inputs.len() - 1) {
        let din = (inputs[i + 1] - inputs[i]) as f64;
        let dout = (outputs[i + 1] - outputs[i]) as f64;
        let slope = (dout / din) * NANO_SCALE_F64;
        if slope > prev_slope + 1e-9 {
            anyhow::bail!(
                "convexity violation between inputs {} and {} (slope {} > prev {})",
                inputs[i],
                inputs[i + 1],
                slope,
                prev_slope
            );
        }
        prev_slope = slope;
    }

    Ok(())
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
