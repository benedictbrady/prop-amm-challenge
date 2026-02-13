use prop_amm_executor::NativeExecutor;
use prop_amm_shared::config::{HyperparameterVariance, SimulationConfig};
use prop_amm_shared::instruction::STORAGE_SIZE;
use prop_amm_shared::nano::{f64_to_nano, nano_to_f64};
use prop_amm_shared::normalizer::{
    after_swap as normalizer_after_swap, compute_swap as normalizer_swap,
};

const EMPTY_STORAGE: [u8; STORAGE_SIZE] = [0u8; STORAGE_SIZE];

fn starter_swap(data: &[u8]) -> u64 {
    if data.len() < 25 {
        return 0;
    }

    let side = data[0];
    let input_amount =
        u64::from_le_bytes(data[1..9].try_into().expect("starter input amount")) as u128;
    let reserve_x = u64::from_le_bytes(data[9..17].try_into().expect("starter reserve x")) as u128;
    let reserve_y = u64::from_le_bytes(data[17..25].try_into().expect("starter reserve y")) as u128;

    if reserve_x == 0 || reserve_y == 0 {
        return 0;
    }

    let k = reserve_x.saturating_mul(reserve_y);
    match side {
        0 => {
            let net_y = input_amount.saturating_mul(950) / 1000;
            let new_ry = reserve_y + net_y;
            reserve_x.saturating_sub((k + new_ry - 1) / new_ry) as u64
        }
        1 => {
            let net_x = input_amount.saturating_mul(950) / 1000;
            let new_rx = reserve_x + net_x;
            reserve_y.saturating_sub((k + new_rx - 1) / new_rx) as u64
        }
        _ => 0,
    }
}

fn starter_after_swap(_data: &[u8], _storage: &mut [u8]) {
    // No-op: starter strategy doesn't update storage.
}

fn linear_swap(data: &[u8]) -> u64 {
    if data.len() < 25 {
        return 0;
    }

    let input_amount = u64::from_le_bytes(data[1..9].try_into().expect("linear input amount"));
    input_amount
}

fn normalizer_exec() -> NativeExecutor {
    NativeExecutor::new(normalizer_swap, Some(normalizer_after_swap))
}

fn starter_exec() -> NativeExecutor {
    NativeExecutor::new(starter_swap, Some(starter_after_swap))
}

#[test]
fn test_normalizer_basic_execution() {
    let exec = normalizer_exec();

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let output = exec.execute(0, f64_to_nano(10.0), rx, ry, &EMPTY_STORAGE);
    let output_f64 = nano_to_f64(output);
    assert!(
        output_f64 > 0.09 && output_f64 < 0.11,
        "buy output: {}",
        output_f64
    );

    let output = exec.execute(1, f64_to_nano(1.0), rx, ry, &EMPTY_STORAGE);
    let output_f64 = nano_to_f64(output);
    assert!(
        output_f64 > 95.0 && output_f64 < 100.0,
        "sell output: {}",
        output_f64
    );
}

#[test]
fn test_normalizer_math_correctness() {
    let exec = normalizer_exec();

    let output = exec.execute(
        0,
        f64_to_nano(100.0),
        f64_to_nano(100.0),
        f64_to_nano(10000.0),
        &EMPTY_STORAGE,
    );
    let output_f64 = nano_to_f64(output);
    assert!(
        (output_f64 - 0.987).abs() < 0.01,
        "expected ~0.987, got {}",
        output_f64
    );
}

#[test]
fn test_starter_has_higher_fee() {
    let exec_norm = normalizer_exec();
    let exec_start = starter_exec();

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let input = f64_to_nano(50.0);

    let norm_out = exec_norm.execute(0, input, rx, ry, &EMPTY_STORAGE);
    let start_out = exec_start.execute(0, input, rx, ry, &EMPTY_STORAGE);
    assert!(
        norm_out > start_out,
        "normalizer ({}) should beat starter ({})",
        norm_out,
        start_out
    );
}

#[test]
fn test_monotonicity() {
    let exec = normalizer_exec();

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let sizes = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0];
    let mut prev = 0u64;
    for &size in &sizes {
        let out = exec.execute(0, f64_to_nano(size), rx, ry, &EMPTY_STORAGE);
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
    let exec = normalizer_exec();

    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);

    let sizes = [1.0, 10.0, 50.0, 100.0, 500.0];
    let eps = 0.001;
    let mut prev_marginal = f64::MAX;

    for &size in &sizes {
        let out_lo = nano_to_f64(exec.execute(0, f64_to_nano(size), rx, ry, &EMPTY_STORAGE));
        let out_hi = nano_to_f64(exec.execute(0, f64_to_nano(size + eps), rx, ry, &EMPTY_STORAGE));
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
fn test_legacy_float_concavity_check_false_positive_for_linear_compute_swap() {
    // This swap function is affine in input amount (y = x), so it is both monotone and concave.
    // A legacy float-based finite-difference check can still fail because f64_to_nano(size + eps)
    // is not always exactly +1_000_000 nanos larger than f64_to_nano(size).
    let exec = NativeExecutor::new(linear_swap, None);
    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let storage = [0u8; STORAGE_SIZE];

    let trade_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
    let eps = 0.001;

    // Monotonicity in the same style as validate.rs.
    let mut prev_output = 0u64;
    for &size in &trade_sizes {
        let output = exec.execute(0, f64_to_nano(size), rx, ry, &storage);
        assert!(
            output > prev_output || prev_output == 0,
            "linear swap should be monotone at size {} ({} <= {})",
            size,
            output,
            prev_output
        );
        prev_output = output;
    }

    // Show the truncation discontinuity that causes false-positive concavity failure.
    assert_eq!(
        f64_to_nano(1.0 + eps) - f64_to_nano(1.0),
        999_999,
        "expected 1.0 -> 1.001 nano delta to truncate to 999,999"
    );
    assert_eq!(
        f64_to_nano(2.0 + eps) - f64_to_nano(2.0),
        1_000_000,
        "expected 2.0 -> 2.001 nano delta to truncate to 1,000,000"
    );

    // Concavity logic copied from validate.rs should trip despite linear shape.
    let mut prev_marginal = f64::MAX;
    let mut violation: Option<(f64, f64, f64)> = None;
    for &size in &trade_sizes {
        let out_lo = nano_to_f64(exec.execute(0, f64_to_nano(size), rx, ry, &storage));
        let out_hi = nano_to_f64(exec.execute(0, f64_to_nano(size + eps), rx, ry, &storage));
        let marginal = (out_hi - out_lo) / eps;
        if marginal > prev_marginal + 1e-9 {
            violation = Some((size, prev_marginal, marginal));
            break;
        }
        prev_marginal = marginal;
    }

    let (size, prev, current) = violation.expect(
        "expected validator-style concavity check to false-positive on linear compute_swap",
    );
    assert!(
        (size - 2.0).abs() < 1e-12,
        "expected first violation at size 2.0, got size={} (prev={}, current={})",
        size,
        prev,
        current
    );
}

#[test]
fn test_integer_concavity_check_accepts_linear_compute_swap() {
    let exec = NativeExecutor::new(linear_swap, None);
    let rx = f64_to_nano(100.0);
    let ry = f64_to_nano(10000.0);
    let storage = [0u8; STORAGE_SIZE];
    let trade_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
    let delta_nano = 1_000_000u64;

    for &size in &trade_sizes {
        let in_0 = f64_to_nano(size);
        let in_1 = in_0 + delta_nano;
        let in_2 = in_1 + delta_nano;

        let out_0 = exec.execute(0, in_0, rx, ry, &storage) as i128;
        let out_1 = exec.execute(0, in_1, rx, ry, &storage) as i128;
        let out_2 = exec.execute(0, in_2, rx, ry, &storage) as i128;
        let step_1 = out_1 - out_0;
        let step_2 = out_2 - out_1;

        assert!(
            step_2 <= step_1 + 1,
            "integer concavity check should accept linear swap at size {} (step2={} > step1={})",
            size,
            step_2,
            step_1
        );
    }
}

#[test]
fn test_normalizer_vs_normalizer_zero_edge() {
    let config = SimulationConfig {
        n_steps: 500,
        seed: 42,
        ..SimulationConfig::default()
    };
    let result = prop_amm_sim::engine::run_simulation_native(
        normalizer_swap,
        Some(normalizer_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        &config,
    )
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
    let result = prop_amm_sim::engine::run_simulation_native(
        starter_swap,
        Some(starter_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        &config,
    )
    .unwrap();
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

    let result = prop_amm_sim::runner::run_batch_native(
        starter_swap,
        Some(starter_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        configs,
        Some(2),
    )
    .unwrap();
    assert_eq!(result.n_sims(), 4);
}

#[test]
fn test_after_swap_noop() {
    let exec = starter_exec();
    let mut storage = [0u8; STORAGE_SIZE];

    exec.execute_after_swap(0, 1000, 500, 2000, 3000, 0, &mut storage);
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
    let result = prop_amm_sim::engine::run_simulation_native(
        starter_swap,
        Some(starter_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        &config,
    )
    .unwrap();
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
    let result1 = prop_amm_sim::engine::run_simulation_native(
        starter_swap,
        Some(starter_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        &config,
    )
    .unwrap();
    let result2 = prop_amm_sim::engine::run_simulation_native(
        starter_swap,
        Some(starter_after_swap),
        normalizer_swap,
        Some(normalizer_after_swap),
        &config,
    )
    .unwrap();
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
