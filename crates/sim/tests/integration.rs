use prop_amm_executor::{BpfExecutor, BpfProgram};
use prop_amm_shared::config::{HyperparameterVariance, SimulationConfig};
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

    exec.execute_after_swap(0, 1000, 500, 2000, 3000, 0, &mut storage)
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

#[test]
fn test_native_normalizer_fee_from_storage() {
    use prop_amm_shared::normalizer::compute_swap;
    use prop_amm_shared::instruction::encode_swap_instruction;

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let input = f64_to_nano(100.0);

    // Default (zero storage) → 30bps
    let storage_zero = [0u8; STORAGE_SIZE];
    let data_zero = encode_swap_instruction(0, input, rx, ry, &storage_zero);
    let out_default = compute_swap(&data_zero);

    // Explicit 30bps → same as default
    let mut storage_30 = [0u8; STORAGE_SIZE];
    storage_30[0..2].copy_from_slice(&30u16.to_le_bytes());
    let data_30 = encode_swap_instruction(0, input, rx, ry, &storage_30);
    let out_30 = compute_swap(&data_30);
    assert_eq!(out_default, out_30, "zero storage should equal explicit 30bps");

    // 100bps (1%) → less output than 30bps
    let mut storage_100 = [0u8; STORAGE_SIZE];
    storage_100[0..2].copy_from_slice(&100u16.to_le_bytes());
    let data_100 = encode_swap_instruction(0, input, rx, ry, &storage_100);
    let out_100 = compute_swap(&data_100);
    assert!(out_100 < out_30, "100bps ({}) should give less output than 30bps ({})", out_100, out_30);

    // 10bps → more output than 30bps
    let mut storage_10 = [0u8; STORAGE_SIZE];
    storage_10[0..2].copy_from_slice(&10u16.to_le_bytes());
    let data_10 = encode_swap_instruction(0, input, rx, ry, &storage_10);
    let out_10 = compute_swap(&data_10);
    assert!(out_10 > out_30, "10bps ({}) should give more output than 30bps ({})", out_10, out_30);
}

#[test]
fn test_norm_liquidity_mult_affects_edge() {
    use prop_amm_shared::normalizer::{compute_swap as norm_swap, after_swap as norm_after};

    // Low liquidity normalizer (0.5x) — easier to beat
    let config_low = SimulationConfig {
        n_steps: 1000,
        seed: 42,
        norm_liquidity_mult: 0.5,
        ..SimulationConfig::default()
    };
    let result_low = prop_amm_sim::engine::run_simulation_native(
        norm_swap, Some(norm_after), norm_swap, Some(norm_after), &config_low,
    ).unwrap();

    // High liquidity normalizer (2.0x) — harder to beat
    let config_high = SimulationConfig {
        n_steps: 1000,
        seed: 42,
        norm_liquidity_mult: 2.0,
        ..SimulationConfig::default()
    };
    let result_high = prop_amm_sim::engine::run_simulation_native(
        norm_swap, Some(norm_after), norm_swap, Some(norm_after), &config_high,
    ).unwrap();

    // Different liquidity should produce different edges
    assert!(
        (result_low.submission_edge - result_high.submission_edge).abs() > 0.01,
        "different liquidity mults should produce different edges: low={}, high={}",
        result_low.submission_edge, result_high.submission_edge
    );
}

#[test]
fn test_hyperparameter_variance_generates_varied_configs() {
    let variance = HyperparameterVariance::default();
    let configs = variance.generate_configs(100);

    assert_eq!(configs.len(), 100);

    let sigma_min = configs.iter().map(|c| c.gbm_sigma).fold(f64::INFINITY, f64::min);
    let sigma_max = configs.iter().map(|c| c.gbm_sigma).fold(f64::NEG_INFINITY, f64::max);
    assert!(sigma_min >= 0.0001, "sigma_min {} below range", sigma_min);
    assert!(sigma_max <= 0.007, "sigma_max {} above range", sigma_max);
    assert!(
        sigma_max - sigma_min > 0.003,
        "sigma range too narrow: [{}, {}]",
        sigma_min,
        sigma_max
    );

    let fee_min = configs.iter().map(|c| c.norm_fee_bps).min().unwrap();
    let fee_max = configs.iter().map(|c| c.norm_fee_bps).max().unwrap();
    assert!(fee_min >= 30, "fee_min {} below range", fee_min);
    assert!(fee_max <= 80, "fee_max {} above range", fee_max);
    assert!(fee_max - fee_min > 30, "fee range too narrow: [{}, {}]", fee_min, fee_max);

    let liq_min = configs.iter().map(|c| c.norm_liquidity_mult).fold(f64::INFINITY, f64::min);
    let liq_max = configs.iter().map(|c| c.norm_liquidity_mult).fold(f64::NEG_INFINITY, f64::max);
    assert!(liq_min >= 0.4, "liq_min {} below range", liq_min);
    assert!(liq_max <= 2.0, "liq_max {} above range", liq_max);
    assert!(liq_max - liq_min > 0.5, "liq range too narrow: [{}, {}]", liq_min, liq_max);
}
