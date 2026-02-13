use std::path::Path;
use std::sync::atomic::{AtomicPtr, Ordering};

use anyhow::Context;
use prop_amm_executor::{AfterSwapFn, BpfExecutor, BpfProgram};
use prop_amm_shared::instruction::STORAGE_SIZE;
use prop_amm_shared::nano::{f64_to_nano, nano_to_f64};
use prop_amm_shared::normalizer::{
    after_swap as normalizer_after_swap, compute_swap as normalizer_swap,
};
use prop_amm_sim::runner;
use syn::{Expr, Item, Lit, Type};

use super::compile;

type FfiSwapFn = unsafe extern "C" fn(*const u8, usize) -> u64;
type FfiAfterSwapFn = unsafe extern "C" fn(*const u8, usize, *mut u8, usize);

static LOADED_SWAP: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static LOADED_AFTER_SWAP: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

const PARITY_SIMS: u32 = 12;
const PARITY_STEPS: u32 = 2_000;
const PARITY_SEED_START: u64 = 9_001;
const PARITY_SEED_STRIDE: u64 = 7;
const PARITY_ABS_TOL: f64 = 1e-6;
const CONCAVITY_DELTA_NANO: u64 = 1_000_000;
const CONCAVITY_STEP_TOL_NANO: i128 = 1;

fn dynamic_swap(data: &[u8]) -> u64 {
    let ptr = LOADED_SWAP.load(Ordering::Relaxed);
    let f: FfiSwapFn = unsafe { std::mem::transmute(ptr) };
    unsafe { f(data.as_ptr(), data.len()) }
}

fn dynamic_after_swap(data: &[u8], storage: &mut [u8]) {
    let ptr = LOADED_AFTER_SWAP.load(Ordering::Relaxed);
    let f: FfiAfterSwapFn = unsafe { std::mem::transmute(ptr) };
    unsafe {
        f(
            data.as_ptr(),
            data.len(),
            storage.as_mut_ptr(),
            storage.len(),
        )
    }
}

pub fn run(file: &str) -> anyhow::Result<()> {
    let metadata = validate_submission_metadata(file)?;
    println!("  [PASS] Name: {}", metadata.name);
    if metadata.model_used == "None" {
        println!("  [PASS] Model used: None (human-written)");
    } else {
        println!("  [PASS] Model used: {}", metadata.model_used);
    }

    println!("Compiling {} (BPF)...", file);
    let so_path = compile::compile_bpf(file)?;
    println!("Compiling {} (native)...", file);
    let native_path = compile::compile_native(file)?;

    println!("Validating program: {}", so_path.display());

    // Load ELF
    let elf_bytes = std::fs::read(&so_path)?;
    let program =
        BpfProgram::load(&elf_bytes).map_err(|e| anyhow::anyhow!("Failed to load ELF: {}", e))?;
    println!("  [PASS] ELF loaded and verified");

    let parity_program = program.clone();
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

    // Concavity check: marginal output per additional input must not increase.
    println!("  Checking concavity...");
    // Buy side concavity: for fixed nano step Δ, discrete marginal output should not increase.
    for &size in &trade_sizes {
        let in_0 = f64_to_nano(size);
        let in_1 = in_0.saturating_add(CONCAVITY_DELTA_NANO);
        let in_2 = in_1.saturating_add(CONCAVITY_DELTA_NANO);

        if in_1 <= in_0 || in_2 <= in_1 {
            continue;
        }

        let out_0 = executor.execute(0, in_0, rx, ry, &storage)? as i128;
        let out_1 = executor.execute(0, in_1, rx, ry, &storage)? as i128;
        let out_2 = executor.execute(0, in_2, rx, ry, &storage)? as i128;
        let step_1 = out_1 - out_0;
        let step_2 = out_2 - out_1;

        if step_2 > step_1 + CONCAVITY_STEP_TOL_NANO {
            anyhow::bail!(
                "FAIL: Concavity violation (buy side). At size={}, step2={} > step1={} (delta={} nanos)",
                size,
                step_2,
                step_1,
                CONCAVITY_DELTA_NANO
            );
        }
    }
    println!("  [PASS] Buy side concavity");

    // Sell side concavity: for fixed nano step Δ, discrete marginal output should not increase.
    for &size in &trade_sizes {
        let in_0 = f64_to_nano(size);
        let in_1 = in_0.saturating_add(CONCAVITY_DELTA_NANO);
        let in_2 = in_1.saturating_add(CONCAVITY_DELTA_NANO);

        if in_1 <= in_0 || in_2 <= in_1 {
            continue;
        }

        let out_0 = executor.execute(1, in_0, rx, ry, &storage)? as i128;
        let out_1 = executor.execute(1, in_1, rx, ry, &storage)? as i128;
        let out_2 = executor.execute(1, in_2, rx, ry, &storage)? as i128;
        let step_1 = out_1 - out_0;
        let step_2 = out_2 - out_1;

        if step_2 > step_1 + CONCAVITY_STEP_TOL_NANO {
            anyhow::bail!(
                "FAIL: Concavity violation (sell side). At size={}, step2={} > step1={} (delta={} nanos)",
                size,
                step_2,
                step_1,
                CONCAVITY_DELTA_NANO
            );
        }
    }
    println!("  [PASS] Sell side concavity");

    // Randomized behavioral checks over varied reserve/storage states
    println!("  Checking randomized reserve/storage states...");
    for seed in 0..32u64 {
        let mut storage = [0u8; STORAGE_SIZE];
        for i in 0..32usize {
            storage[i] = (mix(seed.wrapping_add(i as u64)) & 0xFF) as u8;
        }

        let rx = 1_000_000_000u64 + (mix(seed ^ 0x0123_4567_89AB_CDEF) % 2_000_000_000_000u64);
        let ry = 1_000_000_000u64 + (mix(seed ^ 0x0F0F_0F0F_F0F0_F0F0) % 200_000_000_000_000u64);

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
    }
    println!("  [PASS] Randomized reserve/storage checks");

    run_native_bpf_parity_check(parity_program, &native_path)?;

    println!("\nAll validation checks passed!");
    Ok(())
}

fn run_native_bpf_parity_check(program: BpfProgram, native_path: &Path) -> anyhow::Result<()> {
    println!(
        "  Checking native/BPF parity ({} sims, {} steps, seeds {} + i*{})...",
        PARITY_SIMS, PARITY_STEPS, PARITY_SEED_START, PARITY_SEED_STRIDE
    );

    let submission_after_swap = load_native_submission(native_path)?;

    let native = runner::run_default_batch_native_seeded(
        dynamic_swap,
        submission_after_swap,
        normalizer_swap,
        Some(normalizer_after_swap),
        PARITY_SIMS,
        PARITY_STEPS,
        Some(4),
        PARITY_SEED_START,
        PARITY_SEED_STRIDE,
    )?;
    let bpf = runner::run_default_batch_mixed_seeded(
        program,
        normalizer_swap,
        Some(normalizer_after_swap),
        PARITY_SIMS,
        PARITY_STEPS,
        Some(4),
        PARITY_SEED_START,
        PARITY_SEED_STRIDE,
    )?;

    let total_delta = (native.total_edge - bpf.total_edge).abs();
    let avg_delta = (native.avg_edge() - bpf.avg_edge()).abs();

    println!(
        "    native_total={:.9} bpf_total={:.9} delta={:.9} tol={:.9}",
        native.total_edge, bpf.total_edge, total_delta, PARITY_ABS_TOL
    );
    println!(
        "    native_avg={:.9} bpf_avg={:.9} delta={:.9} tol={:.9}",
        native.avg_edge(),
        bpf.avg_edge(),
        avg_delta,
        PARITY_ABS_TOL
    );

    if total_delta > PARITY_ABS_TOL || avg_delta > PARITY_ABS_TOL {
        anyhow::bail!(
            "FAIL: Native/BPF parity check failed. avg_delta={:.9}, total_delta={:.9}, tol={:.9}",
            avg_delta,
            total_delta,
            PARITY_ABS_TOL
        );
    }

    println!("  [PASS] Native/BPF parity");
    Ok(())
}

fn load_native_submission(native_path: &Path) -> anyhow::Result<Option<AfterSwapFn>> {
    let lib = Box::new(
        unsafe { libloading::Library::new(native_path) }.map_err(|e| {
            anyhow::anyhow!(
                "Failed to load native library {}: {}",
                native_path.display(),
                e
            )
        })?,
    );
    let lib = Box::leak(lib);

    let swap_fn: libloading::Symbol<FfiSwapFn> = unsafe {
        lib.get(compile::NATIVE_SWAP_SYMBOL)
            .or_else(|_| lib.get(b"compute_swap_ffi"))
    }
    .map_err(|e| anyhow::anyhow!("Missing native swap symbol: {}", e))?;
    LOADED_SWAP.store(*swap_fn as *mut (), Ordering::Relaxed);

    let has_after_swap = if let Ok(after_fn) = unsafe {
        lib.get::<FfiAfterSwapFn>(compile::NATIVE_AFTER_SWAP_SYMBOL)
            .or_else(|_| lib.get::<FfiAfterSwapFn>(b"after_swap_ffi"))
    } {
        LOADED_AFTER_SWAP.store(*after_fn as *mut (), Ordering::Relaxed);
        true
    } else {
        false
    };

    Ok(if has_after_swap {
        Some(dynamic_after_swap)
    } else {
        None
    })
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

struct SubmissionMetadata {
    name: String,
    model_used: String,
}

fn validate_submission_metadata(file: &str) -> anyhow::Result<SubmissionMetadata> {
    let source = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read {} for metadata checks: {}", file, e))?;
    let parsed = syn::parse_file(&source)
        .map_err(|e| anyhow::anyhow!("Failed to parse {} for metadata checks: {}", file, e))?;

    let mut name: Option<String> = None;
    let mut model_used: Option<String> = None;
    let mut has_get_model_used = false;

    for item in parsed.items {
        match item {
            Item::Const(item_const) => {
                let ident = item_const.ident.to_string();
                if ident == "NAME" {
                    name = extract_str_const(&item_const.ty, &item_const.expr)
                        .context("`NAME` must be a string literal constant")?;
                } else if ident == "MODEL_USED" {
                    model_used = extract_str_const(&item_const.ty, &item_const.expr)
                        .context("`MODEL_USED` must be a string literal constant")?;
                }
            }
            Item::Fn(item_fn) => {
                if item_fn.sig.ident == "get_model_used" {
                    has_get_model_used = true;
                }
            }
            _ => {}
        }
    }

    let name = name
        .ok_or_else(|| anyhow::anyhow!("Submission must define `const NAME: &str = \"...\";`"))?;
    if name.trim().is_empty() {
        anyhow::bail!("`NAME` must not be empty");
    }

    let model_used = model_used.ok_or_else(|| {
        anyhow::anyhow!("Submission must define `const MODEL_USED: &str = \"...\";`")
    })?;
    if model_used.trim().is_empty() {
        anyhow::bail!(
            "`MODEL_USED` must not be empty. Use \"None\" for human-written submissions."
        );
    }

    if !has_get_model_used {
        anyhow::bail!("Submission must define `fn get_model_used() -> &'static str`");
    }

    Ok(SubmissionMetadata { name, model_used })
}

fn extract_str_const(ty: &Type, expr: &Expr) -> anyhow::Result<Option<String>> {
    let is_str_ref = match ty {
        Type::Reference(r) => match &*r.elem {
            Type::Path(p) => p.path.is_ident("str"),
            _ => false,
        },
        _ => false,
    };
    if !is_str_ref {
        return Ok(None);
    }

    match expr {
        Expr::Lit(expr_lit) => match &expr_lit.lit {
            Lit::Str(s) => Ok(Some(s.value())),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}
