use std::sync::atomic::{AtomicPtr, Ordering};

use prop_amm_executor::{AfterSwapFn, BpfProgram};
use prop_amm_shared::normalizer::{
    after_swap as normalizer_after_swap_fn, compute_swap as normalizer_swap,
};
use prop_amm_sim::runner;

use super::compile;
use crate::output;

type FfiSwapFn = unsafe extern "C" fn(*const u8, usize) -> u64;
type FfiAfterSwapFn = unsafe extern "C" fn(*const u8, usize, *mut u8, usize);

static LOADED_SWAP: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static LOADED_AFTER_SWAP: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

fn dynamic_swap(data: &[u8]) -> u64 {
    let ptr = LOADED_SWAP.load(Ordering::Relaxed);
    let f: FfiSwapFn = unsafe { std::mem::transmute(ptr) };
    unsafe { f(data.as_ptr(), data.len()) }
}

fn dynamic_after_swap(data: &[u8], storage: &mut [u8]) {
    let ptr = LOADED_AFTER_SWAP.load(Ordering::Relaxed);
    let f: FfiAfterSwapFn = unsafe { std::mem::transmute(ptr) };
    unsafe { f(data.as_ptr(), data.len(), storage.as_mut_ptr(), storage.len()) }
}

pub fn run(
    file: &str,
    simulations: u32,
    steps: u32,
    workers: usize,
    bpf: bool,
) -> anyhow::Result<()> {
    let n_workers = if workers == 0 { None } else { Some(workers) };

    if bpf {
        run_bpf(file, simulations, steps, n_workers)
    } else {
        run_native(file, simulations, steps, n_workers)
    }
}

fn run_native(
    file: &str,
    simulations: u32,
    steps: u32,
    n_workers: Option<usize>,
) -> anyhow::Result<()> {
    println!("Compiling {} (native)...", file);
    let native_path = compile::compile_native(file)?;

    // Load the native library — leak it so symbols remain valid for the process lifetime.
    let lib = Box::new(
        unsafe { libloading::Library::new(&native_path) }
            .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", native_path.display(), e))?,
    );
    let lib = Box::leak(lib);

    let swap_fn: libloading::Symbol<FfiSwapFn> = unsafe { lib.get(b"compute_swap_ffi") }
        .map_err(|e| anyhow::anyhow!("Missing compute_swap_ffi symbol: {}", e))?;
    LOADED_SWAP.store(*swap_fn as *mut (), Ordering::Relaxed);

    let has_after_swap =
        if let Ok(after_fn) = unsafe { lib.get::<FfiAfterSwapFn>(b"after_swap_ffi") } {
            LOADED_AFTER_SWAP.store(*after_fn as *mut (), Ordering::Relaxed);
            true
        } else {
            false
        };

    let submission_after_swap: Option<AfterSwapFn> = if has_after_swap {
        Some(dynamic_after_swap)
    } else {
        None
    };

    println!(
        "Running {} simulations ({} steps each) natively...",
        simulations, steps,
    );

    let start = std::time::Instant::now();
    let result = runner::run_default_batch_native(
        dynamic_swap,
        submission_after_swap,
        normalizer_swap,
        Some(normalizer_after_swap_fn),
        simulations,
        steps,
        n_workers,
    )?;
    let elapsed = start.elapsed();

    output::print_results(&result, elapsed);
    Ok(())
}

fn run_bpf(
    file: &str,
    simulations: u32,
    steps: u32,
    n_workers: Option<usize>,
) -> anyhow::Result<()> {
    println!("Compiling {} (BPF)...", file);
    let bpf_path = compile::compile_bpf(file)?;

    let bytes = std::fs::read(&bpf_path)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", bpf_path.display(), e))?;
    let submission_program = BpfProgram::load(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load BPF program: {}", e))?;

    println!(
        "Running {} simulations ({} steps each) via BPF{}...",
        simulations,
        steps,
        if submission_program.jit_available() {
            " (JIT)"
        } else {
            " (interpreter)"
        },
    );

    let start = std::time::Instant::now();
    let result = runner::run_default_batch_mixed(
        submission_program,
        normalizer_swap,
        Some(normalizer_after_swap_fn),
        simulations,
        steps,
        n_workers,
    )?;
    let elapsed = start.elapsed();

    output::print_results(&result, elapsed);
    Ok(())
}
