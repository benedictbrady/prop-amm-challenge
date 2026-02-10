use prop_amm_executor::{BpfExecutor, BpfProgram, NativeExecutor};
use prop_amm_shared::instruction::STORAGE_SIZE;
use prop_amm_shared::nano::f64_to_nano;
use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
use std::time::Instant;

const NORMALIZER_SO: &[u8] =
    include_bytes!("../../../programs/normalizer/target/deploy/normalizer.so");

pub fn run_profile() {
    let program = BpfProgram::load(NORMALIZER_SO).expect("load");
    let mut bpf_exec = BpfExecutor::new(program.clone());
    let native_exec = NativeExecutor::new(normalizer_swap, None);

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let amount = f64_to_nano(10.0);
    let storage = [0u8; STORAGE_SIZE];

    // Warmup
    for _ in 0..100 {
        let _ = bpf_exec.execute(0, amount, rx, ry, &storage);
    }

    let n = 10_000;

    // BPF benchmark
    let start = Instant::now();
    for _ in 0..n {
        let _ = bpf_exec.execute(0, amount, rx, ry, &storage);
    }
    let bpf_elapsed = start.elapsed();
    let bpf_us = bpf_elapsed.as_micros() as f64 / n as f64;
    println!("=== Per-call Benchmark ===");
    println!(
        "BPF:    {:.1}µs/call ({:.0} calls/sec)",
        bpf_us,
        1_000_000.0 / bpf_us
    );

    // Native benchmark
    let start = Instant::now();
    for _ in 0..n {
        let _ = native_exec.execute(0, amount, rx, ry, &storage);
    }
    let native_elapsed = start.elapsed();
    let native_us = native_elapsed.as_nanos() as f64 / n as f64 / 1000.0;
    println!(
        "Native: {:.3}µs/call ({:.0} calls/sec)",
        native_us,
        1_000_000.0 / native_us
    );
    println!("Speedup: {:.0}x", bpf_us / native_us);

    // Full sim benchmarks
    let config = prop_amm_shared::config::SimulationConfig {
        n_steps: 1000,
        seed: 42,
        ..Default::default()
    };

    // BPF sim
    let p1 = BpfProgram::load(NORMALIZER_SO).expect("load");
    let p2 = BpfProgram::load(NORMALIZER_SO).expect("load");
    let start = Instant::now();
    let _ = crate::engine::run_simulation(p1, p2, &config);
    let bpf_sim = start.elapsed();

    // Native sim
    let start = Instant::now();
    let _ =
        crate::engine::run_simulation_native(normalizer_swap, None, normalizer_swap, None, &config);
    let native_sim = start.elapsed();

    // Mixed sim (BPF submission + native normalizer)
    let p1 = BpfProgram::load(NORMALIZER_SO).expect("load");
    let start = Instant::now();
    let _ = crate::engine::run_simulation_mixed(p1, normalizer_swap, None, &config);
    let mixed_sim = start.elapsed();

    println!("\n=== 1k-step Sim Benchmark ===");
    println!("BPF+BPF:       {:.3}s", bpf_sim.as_secs_f64());
    println!("BPF+Native:    {:.3}s", mixed_sim.as_secs_f64());
    println!("Native+Native: {:.3}s", native_sim.as_secs_f64());

    println!("\n=== 1000-sim / 10k-step Projections (8 workers) ===");
    let bpf_proj = bpf_sim.as_secs_f64() * 10.0 * 1000.0 / 8.0;
    let mixed_proj = mixed_sim.as_secs_f64() * 10.0 * 1000.0 / 8.0;
    let native_proj = native_sim.as_secs_f64() * 10.0 * 1000.0 / 8.0;
    println!("BPF+BPF:       {:.0}s", bpf_proj);
    println!("BPF+Native:    {:.0}s", mixed_proj);
    println!("Native+Native: {:.0}s", native_proj);
}
