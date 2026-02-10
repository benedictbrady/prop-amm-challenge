use prop_amm_executor::{BpfExecutor, BpfProgram};
use prop_amm_shared::config::SimulationConfig;
use prop_amm_shared::instruction::STORAGE_SIZE;
use prop_amm_shared::nano::{f64_to_nano, nano_to_f64};

const NORMALIZER_SO: &[u8] =
    include_bytes!("../../../programs/normalizer/target/deploy/normalizer.so");
const STARTER_SO: &[u8] = include_bytes!("../../../programs/starter/target/deploy/starter.so");

fn load_normalizer() -> BpfProgram {
    BpfProgram::load(NORMALIZER_SO).expect("Failed to load normalizer")
}

fn load_starter() -> BpfProgram {
    BpfProgram::load(STARTER_SO).expect("Failed to load starter")
}

const EMPTY_STORAGE: [u8; STORAGE_SIZE] = [0u8; STORAGE_SIZE];

#[test]
fn test_normalizer_basic_execution() {
    let program = load_normalizer();
    let mut exec = BpfExecutor::new(program);

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let output = exec
        .execute(0, f64_to_nano(10.0), rx, ry, &EMPTY_STORAGE)
        .unwrap();
    let output_f64 = nano_to_f64(output);
    assert!(
        output_f64 > 0.09 && output_f64 < 0.11,
        "buy output: {}",
        output_f64
    );

    let output = exec
        .execute(1, f64_to_nano(1.0), rx, ry, &EMPTY_STORAGE)
        .unwrap();
    let output_f64 = nano_to_f64(output);
    assert!(
        output_f64 > 95.0 && output_f64 < 100.0,
        "sell output: {}",
        output_f64
    );
}

#[test]
fn test_normalizer_math_correctness() {
    let program = load_normalizer();
    let mut exec = BpfExecutor::new(program);

    let output = exec
        .execute(
            0,
            f64_to_nano(100.0),
            f64_to_nano(100.0),
            f64_to_nano(10000.0),
            &EMPTY_STORAGE,
        )
        .unwrap();
    let output_f64 = nano_to_f64(output);
    assert!(
        (output_f64 - 0.987).abs() < 0.01,
        "expected ~0.987, got {}",
        output_f64
    );
}

#[test]
fn test_starter_has_higher_fee() {
    let mut exec_norm = BpfExecutor::new(load_normalizer());
    let mut exec_start = BpfExecutor::new(load_starter());

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let input = f64_to_nano(50.0);

    let norm_out = exec_norm.execute(0, input, rx, ry, &EMPTY_STORAGE).unwrap();
    let start_out = exec_start
        .execute(0, input, rx, ry, &EMPTY_STORAGE)
        .unwrap();
    assert!(
        norm_out > start_out,
        "normalizer ({}) should beat starter ({})",
        norm_out,
        start_out
    );
}

#[test]
fn test_monotonicity() {
    let mut exec = BpfExecutor::new(load_normalizer());

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let sizes = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0];
    let mut prev = 0u64;
    for &size in &sizes {
        let out = exec
            .execute(0, f64_to_nano(size), rx, ry, &EMPTY_STORAGE)
            .unwrap();
        assert!(
            out > prev,
            "monotonicity violated at size {}: {} <= {}",
            size,
            out,
            prev
        );
        prev = out;
    }
}

#[test]
fn test_convexity() {
    let mut exec = BpfExecutor::new(load_normalizer());

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let sizes = [1.0, 10.0, 50.0, 100.0, 500.0];
    let eps = 0.001;
    let mut prev_marginal = f64::MAX;

    for &size in &sizes {
        let out_lo = nano_to_f64(
            exec.execute(0, f64_to_nano(size), rx, ry, &EMPTY_STORAGE)
                .unwrap(),
        );
        let out_hi = nano_to_f64(
            exec.execute(0, f64_to_nano(size + eps), rx, ry, &EMPTY_STORAGE)
                .unwrap(),
        );
        let marginal = (out_hi - out_lo) / eps;
        assert!(
            marginal <= prev_marginal + 1e-9,
            "convexity violated at size {}",
            size
        );
        prev_marginal = marginal;
    }
}

#[test]
fn test_normalizer_vs_normalizer_zero_edge() {
    let config = SimulationConfig {
        n_steps: 500,
        seed: 42,
        ..SimulationConfig::default()
    };
    let result =
        prop_amm_sim::engine::run_simulation(load_normalizer(), load_normalizer(), &config)
            .unwrap();
    assert!(
        result.submission_edge.abs() < 50.0,
        "edge should be ~0, got {}",
        result.submission_edge
    );
}

#[test]
fn test_simulation_produces_positive_edge() {
    // Any reasonable CFMM should produce positive edge (retail spread > arb loss)
    let config = SimulationConfig {
        n_steps: 2000,
        seed: 42,
        ..SimulationConfig::default()
    };
    let result =
        prop_amm_sim::engine::run_simulation(load_starter(), load_normalizer(), &config).unwrap();
    assert!(
        result.submission_edge > 0.0,
        "submission edge should be positive, got {}",
        result.submission_edge
    );
}

#[test]
fn test_batch_runner() {
    let configs: Vec<SimulationConfig> = (0..4)
        .map(|i| SimulationConfig {
            n_steps: 500,
            seed: i,
            ..SimulationConfig::default()
        })
        .collect();

    let result =
        prop_amm_sim::runner::run_batch(load_starter(), load_normalizer(), configs, Some(2))
            .unwrap();
    assert_eq!(result.n_sims(), 4);
}

#[test]
fn test_after_swap_noop() {
    // Verify after_swap with tag=2 doesn't crash for starter program
    let program = load_starter();
    let mut exec = BpfExecutor::new(program);
    let mut storage = [0u8; STORAGE_SIZE];

    exec.execute_after_swap(0, 1000, 500, 2000, 3000, &mut storage)
        .unwrap();
    // Storage should remain unchanged (starter is a no-op)
    assert_eq!(storage, [0u8; STORAGE_SIZE]);
}

#[test]
fn test_storage_persists_across_swaps() {
    // Use the simulation engine to verify storage flows through correctly.
    // Since starter/normalizer don't use storage, just verify it doesn't crash
    // and that the engine runs with the new storage-enabled paths.
    let config = SimulationConfig {
        n_steps: 100,
        seed: 99,
        ..SimulationConfig::default()
    };
    let result =
        prop_amm_sim::engine::run_simulation(load_starter(), load_normalizer(), &config).unwrap();
    // Just verify it completed without error
    assert!(result.submission_edge.is_finite(), "edge should be finite");
}

#[test]
fn test_storage_reset_between_simulations() {
    // Run two simulations with the same config — they should produce identical results
    // since storage resets between sims
    let config = SimulationConfig {
        n_steps: 500,
        seed: 42,
        ..SimulationConfig::default()
    };
    let result1 =
        prop_amm_sim::engine::run_simulation(load_starter(), load_normalizer(), &config).unwrap();
    let result2 =
        prop_amm_sim::engine::run_simulation(load_starter(), load_normalizer(), &config).unwrap();
    assert_eq!(
        result1.submission_edge, result2.submission_edge,
        "same config should produce identical results when storage resets"
    );
}
