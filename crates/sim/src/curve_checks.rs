use prop_amm_shared::nano::{f64_to_nano, nano_to_f64};

// Finite differences involve two endpoints; allow a few nanos of ambiguity from endpoint
// quantization and round-trip noise.
const QUOTE_DELTA_UNCERTAINTY_NANO: u64 = 4;
// Adjacent sample x-values that differ by <= 4 nanos are effectively the same grid point for
// shape-check purposes.
const INPUT_MERGE_EPS_NANO: u64 = 4;
// Keep runtime shape checks aligned with validator granularity (`CONCAVITY_DELTA_NANO`).
const MIN_CONCAVITY_DX_NANO: u64 = 1_000_000;

pub(crate) fn enforce_submission_monotonic_concave(
    amm_name: &str,
    points: &[(f64, f64)],
    min_input: f64,
    context: &str,
) {
    if amm_name != "submission" {
        return;
    }

    if let Some(message) = submission_shape_violation(points, min_input) {
        panic!("submission shape violation during {context}: {message}");
    }
}

fn submission_shape_violation(points: &[(f64, f64)], min_input: f64) -> Option<String> {
    let min_input_nano = f64_to_nano(min_input);
    let mut sorted: Vec<(u64, u64)> = points
        .iter()
        .copied()
        .filter(|(input, output)| {
            input.is_finite() && output.is_finite() && *input > min_input && *output >= 0.0
        })
        .map(|(input, output)| (f64_to_nano(input), f64_to_nano(output)))
        .filter(|(input, _)| *input > min_input_nano)
        .collect();
    sorted.sort_by_key(|(input, _)| *input);

    let mut cleaned: Vec<(u64, u64)> = Vec::with_capacity(sorted.len());
    for (input, output) in sorted {
        if let Some((prev_input, prev_output)) = cleaned.last_mut() {
            if input.saturating_sub(*prev_input) <= INPUT_MERGE_EPS_NANO {
                if output > *prev_output {
                    *prev_output = output;
                }
                continue;
            }
        }
        cleaned.push((input, output));
    }

    for window in cleaned.windows(2) {
        let (in_a, out_a) = window[0];
        let (in_b, out_b) = window[1];
        if in_b > in_a && out_b.saturating_add(QUOTE_DELTA_UNCERTAINTY_NANO) < out_a {
            return Some(format!(
                "monotonicity violated: input {in_a:.6} -> output {out_a:.6}, \
                 input {in_b:.6} -> output {out_b:.6}",
                in_a = nano_to_f64(in_a),
                out_a = nano_to_f64(out_a),
                in_b = nano_to_f64(in_b),
                out_b = nano_to_f64(out_b),
            ));
        }
    }

    let mut prev_segment: Option<(u64, u64)> = None; // (dy, dx)
    for window in cleaned.windows(2) {
        let (in_a, out_a) = window[0];
        let (in_b, out_b) = window[1];
        let dx = in_b - in_a;
        if dx == 0 {
            continue;
        }
        if dx < MIN_CONCAVITY_DX_NANO {
            continue;
        }
        let dy = out_b.saturating_sub(out_a);
        if let Some((prev_dy, prev_dx)) = prev_segment {
            let prev_upper = prev_dy.saturating_add(QUOTE_DELTA_UNCERTAINTY_NANO);
            let curr_lower = dy.saturating_sub(QUOTE_DELTA_UNCERTAINTY_NANO);
            let lhs = (prev_upper as u128) * (dx as u128);
            let rhs = (curr_lower as u128) * (prev_dx as u128);
            if rhs > lhs {
                let prev_slope = prev_dy as f64 / prev_dx as f64;
                let slope = dy as f64 / dx as f64;
                return Some(format!(
                    "concavity violated: slope rose from {prev:.9} to {slope:.9} \
                     between inputs {in_a:.6} and {in_b:.6}",
                    prev = prev_slope,
                    slope = slope,
                    in_a = nano_to_f64(in_a),
                    in_b = nano_to_f64(in_b),
                ));
            }
        }
        prev_segment = Some((dy, dx));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::submission_shape_violation;
    use crate::amm::BpfAmm;
    use crate::engine;
    use prop_amm_shared::instruction::decode_instruction;
    use prop_amm_shared::config::{HyperparameterVariance, SimulationConfig};
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    const MIN_INPUT: f64 = 1e-3;

    fn assert_valid(points: &[(f64, f64)], context: &str) {
        if let Some(err) = submission_shape_violation(points, MIN_INPUT) {
            panic!("{context}: unexpected shape violation: {err}");
        }
    }

    fn linear_grid(max_input: f64, n: usize) -> Vec<f64> {
        let start = MIN_INPUT * 1.01;
        let span = (max_input - start).max(1e-6);
        (0..n)
            .map(|i| {
                let t = i as f64 / (n.saturating_sub(1).max(1)) as f64;
                start + t * span
            })
            .collect()
    }

    fn geometric_grid(max_input: f64, n: usize) -> Vec<f64> {
        let start = MIN_INPUT * 1.01;
        let ratio = (max_input / start).max(1.0).powf(1.0 / (n.saturating_sub(1).max(1)) as f64);
        (0..n)
            .map(|i| start * ratio.powf(i as f64))
            .collect()
    }

    fn clustered_grid(max_input: f64, n: usize, power: f64) -> Vec<f64> {
        let start = MIN_INPUT * 1.01;
        let span = (max_input - start).max(1e-6);
        (0..n)
            .map(|i| {
                let t = i as f64 / (n.saturating_sub(1).max(1)) as f64;
                start + t.powf(power) * span
            })
            .collect()
    }

    fn duplicated_unsorted_variant(base: &[(f64, f64)], rng: &mut Pcg64) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(base.len() + base.len() / 5 + 8);
        for (idx, (x, y)) in base.iter().copied().enumerate() {
            points.push((x, y));
            if idx % 9 == 0 {
                points.push((x, y));
            }
            if idx % 17 == 0 {
                // Near-equal x and slightly lower y: still legal after cleanup's max-output merge.
                points.push((x + 1e-13 * (1.0 + x.abs()), y * (1.0 - 1e-12)));
            }
        }
        points.shuffle(rng);
        points
    }

    fn assert_curve_variants<F>(label: &str, max_input: f64, f: F, rng: &mut Pcg64) -> usize
    where
        F: Fn(f64) -> f64,
    {
        let grids = [
            linear_grid(max_input, 161),
            geometric_grid(max_input, 161),
            clustered_grid(max_input, 161, 2.4),
            clustered_grid(max_input, 161, 0.45),
        ];

        let mut checks = 0usize;
        for (grid_idx, grid) in grids.iter().enumerate() {
            let base: Vec<(f64, f64)> = grid.iter().map(|x| (*x, f(*x).max(0.0))).collect();
            assert_valid(&base, &format!("{label} grid{grid_idx} sorted"));
            checks += 1;

            let mut reversed = base.clone();
            reversed.reverse();
            assert_valid(&reversed, &format!("{label} grid{grid_idx} reversed"));
            checks += 1;

            let noisy = duplicated_unsorted_variant(&base, rng);
            assert_valid(&noisy, &format!("{label} grid{grid_idx} dup_unsorted"));
            checks += 1;
        }
        checks
    }

    fn build_piecewise_concave_knots(rng: &mut Pcg64, max_input: f64) -> Vec<(f64, f64)> {
        let n_segments = rng.gen_range(8..28);
        let x0 = MIN_INPUT * 1.01;
        let span = (max_input - x0).max(0.5);

        let weights: Vec<f64> = (0..n_segments).map(|_| rng.gen_range(0.2..2.0)).collect();
        let weight_sum: f64 = weights.iter().sum();

        let mut slopes = Vec::with_capacity(n_segments);
        let mut slope = rng.gen_range(0.01..2.5);
        for _ in 0..n_segments {
            slopes.push(slope);
            slope *= rng.gen_range(0.35..0.99);
        }

        let mut knots = Vec::with_capacity(n_segments + 1);
        let mut x = x0;
        let mut y = 0.0;
        knots.push((x, y));
        for idx in 0..n_segments {
            let dx = span * weights[idx] / weight_sum;
            x += dx;
            y += slopes[idx] * dx;
            knots.push((x, y));
        }
        knots
    }

    fn eval_piecewise_linear(knots: &[(f64, f64)], x: f64) -> f64 {
        if x <= knots[0].0 {
            return knots[0].1;
        }
        for window in knots.windows(2) {
            let (x0, y0) = window[0];
            let (x1, y1) = window[1];
            if x <= x1 {
                let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
                return y0 + t * (y1 - y0);
            }
        }
        knots.last().map(|(_, y)| *y).unwrap_or(0.0)
    }

    #[test]
    fn accepts_simple_concave_curve() {
        let points: Vec<(f64, f64)> = (1..120)
            .map(|i| {
                let x = i as f64 * 0.25;
                (x, (1.0 + x).ln())
            })
            .collect();
        assert_valid(&points, "ln(1+x)");
    }

    #[test]
    fn accepts_unsorted_and_duplicate_inputs() {
        let mut points = vec![
            (0.1, 0.0953102),
            (0.2, 0.1823216),
            (0.2, 0.1823216),
            (0.4, 0.3364722),
            (0.8, 0.5877866),
            (1.6, 0.9555114),
            (3.2, 1.4350845),
            (6.4, 2.0014800),
        ];
        points.reverse();
        assert_valid(&points, "unsorted duplicates");
    }

    #[test]
    fn accepts_staircase_from_quantization() {
        let points: Vec<(f64, f64)> = (1..300)
            .map(|i| {
                let x = i as f64 * 0.05;
                let y = ((1.0 + x).ln() * 1_000_000.0).floor() / 1_000_000.0;
                (x, y)
            })
            .collect();
        assert_valid(&points, "quantized staircase");
    }

    #[test]
    fn accepts_extensive_analytic_concave_monotone_family_matrix() {
        let mut rng = Pcg64::seed_from_u64(0xA11CE5EED);
        let mut checks = 0usize;

        for case_idx in 0..360 {
            let max_input = rng.gen_range(0.5..20_000.0);

            let w_log = rng.gen_range(0.05..1.0);
            let w_pow = rng.gen_range(0.05..1.0);
            let w_exp = rng.gen_range(0.05..1.0);
            let w_rat = rng.gen_range(0.05..1.0);
            let w_asinh = rng.gen_range(0.05..1.0);
            let w_sqrt = rng.gen_range(0.05..1.0);
            let w_sum = w_log + w_pow + w_exp + w_rat + w_asinh + w_sqrt;

            let a_log = rng.gen_range(1e-4..20.0);
            let p_pow = rng.gen_range(0.08..0.98);
            let b_pow = rng.gen_range(1e-3..150.0);
            let k_exp = rng.gen_range(1e-4..5.0);
            let b_rat = rng.gen_range(1e-4..200.0);
            let k_asinh = rng.gen_range(1e-4..5.0);
            let b_sqrt = rng.gen_range(1e-3..250.0);
            let linear = rng.gen_range(0.0..0.05);

            checks += assert_curve_variants(
                &format!("analytic blend case {case_idx}"),
                max_input,
                |x| {
                    let log_term = (1.0 + a_log * x).ln();
                    let pow_term = (x + b_pow).powf(p_pow) - b_pow.powf(p_pow);
                    let exp_term = 1.0 - (-k_exp * x).exp();
                    let rat_term = x / (x + b_rat);
                    let asinh_term = (k_asinh * x).asinh();
                    let sqrt_term = (x + b_sqrt).sqrt() - b_sqrt.sqrt();

                    let blended = w_log * log_term
                        + w_pow * pow_term
                        + w_exp * exp_term
                        + w_rat * rat_term
                        + w_asinh * asinh_term
                        + w_sqrt * sqrt_term;
                    (blended / w_sum + linear * x).max(0.0)
                },
                &mut rng,
            );
        }

        assert!(
            checks >= 4_000,
            "expected a large stress matrix, got only {checks} checks"
        );
    }

    #[test]
    fn accepts_extensive_piecewise_linear_concave_monotone_family_matrix() {
        let mut rng = Pcg64::seed_from_u64(0xBADC0FFE);
        let mut checks = 0usize;

        for case_idx in 0..360 {
            let max_input = rng.gen_range(1.0..30_000.0);
            let knots = build_piecewise_concave_knots(&mut rng, max_input);
            checks += assert_curve_variants(
                &format!("piecewise concave case {case_idx}"),
                max_input,
                |x| eval_piecewise_linear(&knots, x),
                &mut rng,
            );
        }

        assert!(
            checks >= 4_000,
            "expected a large stress matrix, got only {checks} checks"
        );
    }

    #[test]
    fn exposes_false_positive_from_cancellation_prone_concave_curve() {
        // f(x) = sqrt(C + x) - sqrt(C) is monotone and concave for C > 0:
        // f'(x) = 1 / (2*sqrt(C+x)) > 0, f''(x) = -1 / (4*(C+x)^(3/2)) < 0.
        // With large C, naive evaluation suffers cancellation and can create flat-then-jump
        // artifacts that trip the discrete slope-rise check.
        let c: f64 = 1e16;
        let xs = [
            0.9628366933867734,
            0.9828747494989979,
            1.0029128056112224,
            1.0229508617234468,
        ];

        let naive_points: Vec<(f64, f64)> = xs
            .iter()
            .map(|x| (*x, (c + *x).sqrt() - c.sqrt()))
            .collect();
        let err = submission_shape_violation(&naive_points, MIN_INPUT).expect(
            "expected checker to flag cancellation-prone evaluation despite legal underlying shape",
        );
        assert!(err.contains("concavity"), "unexpected error: {err}");

        // Equivalent stable form: sqrt(C+x)-sqrt(C) = x / (sqrt(C+x)+sqrt(C)).
        let stable_points: Vec<(f64, f64)> = xs
            .iter()
            .map(|x| (*x, *x / ((c + *x).sqrt() + c.sqrt())))
            .collect();
        assert_valid(
            &stable_points,
            "stable algebraic form of same legal concave/monotone curve",
        );
    }

    #[test]
    fn rejects_non_monotone_curve() {
        let points = vec![(0.1, 1.0), (0.2, 1.1), (0.3, 1.05), (0.4, 1.2)];
        let err = submission_shape_violation(&points, MIN_INPUT).expect("expected violation");
        assert!(err.contains("monotonicity"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_non_concave_curve() {
        let points = vec![(0.1, 0.1), (0.2, 0.18), (0.3, 0.31), (0.4, 0.45)];
        let err = submission_shape_violation(&points, MIN_INPUT).expect("expected violation");
        assert!(err.contains("concavity"), "unexpected error: {err}");
    }

    #[test]
    fn accepts_normalizer_buy_curves_across_random_configs() {
        let mut rng = Pcg64::seed_from_u64(123);
        for case_idx in 0..400 {
            let reserve_x = rng.gen_range(5.0..5_000.0);
            let reserve_y = reserve_x * rng.gen_range(20.0..500.0);
            let mut amm = BpfAmm::new_native(
                normalizer_swap,
                None,
                reserve_x,
                reserve_y,
                "submission".into(),
            );
            let max_input = reserve_y * rng.gen_range(0.05..2.5);

            let mut points = Vec::with_capacity(80);
            for i in 1..=80 {
                let alpha = i as f64 / 80.0;
                let input = MIN_INPUT + alpha * max_input;
                points.push((input, amm.quote_buy_x(input)));
            }
            assert_valid(&points, &format!("normalizer buy case {case_idx}"));
        }
    }

    #[test]
    fn accepts_normalizer_sell_curves_across_random_configs() {
        let mut rng = Pcg64::seed_from_u64(456);
        for case_idx in 0..400 {
            let reserve_x = rng.gen_range(5.0..5_000.0);
            let reserve_y = reserve_x * rng.gen_range(20.0..500.0);
            let mut amm = BpfAmm::new_native(
                normalizer_swap,
                None,
                reserve_x,
                reserve_y,
                "submission".into(),
            );
            let max_input = reserve_x * rng.gen_range(0.05..2.5);

            let mut points = Vec::with_capacity(80);
            for i in 1..=80 {
                let alpha = i as f64 / 80.0;
                let input = MIN_INPUT + alpha * max_input;
                points.push((input, amm.quote_sell_x(input)));
            }
            assert_valid(&points, &format!("normalizer sell case {case_idx}"));
        }
    }

    #[test]
    fn avoids_false_positive_on_cp30_seed12_runtime_path() {
        let mut base = SimulationConfig::default();
        base.n_steps = 10_000;
        let config = HyperparameterVariance::default().apply(&base, 12);

        let result = catch_unwind(AssertUnwindSafe(|| {
            engine::run_simulation_native(normalizer_swap, None, normalizer_swap, None, &config)
                .expect("simulation should complete");
        }));

        assert!(
            result.is_ok(),
            "normalizer-like CP30 submission should not trip runtime shape checks"
        );
    }

    fn cp_out(side: u8, input: u128, rx: u128, ry: u128, gamma_num: u128) -> u128 {
        if input == 0 || rx == 0 || ry == 0 {
            return 0;
        }
        let net = input.saturating_mul(gamma_num) / 10_000;
        let k = rx.saturating_mul(ry);
        match side {
            0 => {
                let new_ry = ry.saturating_add(net);
                rx.saturating_sub((k + new_ry - 1) / new_ry)
            }
            1 => {
                let new_rx = rx.saturating_add(net);
                ry.saturating_sub((k + new_rx - 1) / new_rx)
            }
            _ => 0,
        }
    }

    fn one_div_swap(data: &[u8], base_fee_bps: u128, subsidy_bps: u128, subsidy_cap_y: u128) -> u64 {
        if data.len() < 25 {
            return 0;
        }
        let (side, input_u64, rx_u64, ry_u64) = decode_instruction(data);
        let input = input_u64 as u128;
        let rx = rx_u64 as u128;
        let ry = ry_u64 as u128;
        if input == 0 || rx == 0 || ry == 0 {
            return 0;
        }

        let base = cp_out(side, input, rx, ry, 10_000u128.saturating_sub(base_fee_bps));
        let bonus = match side {
            0 => {
                let cap_x = subsidy_cap_y.saturating_mul(rx) / ry.max(1);
                let num = input.saturating_mul(rx).saturating_mul(subsidy_bps);
                let den = ry.saturating_mul(10_000).max(1);
                (num / den).min(cap_x)
            }
            1 => {
                let num = input.saturating_mul(ry).saturating_mul(subsidy_bps);
                let den = rx.saturating_mul(10_000).max(1);
                (num / den).min(subsidy_cap_y)
            }
            _ => 0,
        };
        let out = base.saturating_add(bonus);
        match side {
            0 => out.min(rx) as u64,
            1 => out.min(ry) as u64,
            _ => 0,
        }
    }

    fn one_div_40_100_020_swap(data: &[u8]) -> u64 {
        one_div_swap(data, 40, 100, 20_000_000)
    }

    fn one_div_30_80_019_swap(data: &[u8]) -> u64 {
        one_div_swap(data, 30, 80, 19_000_000)
    }

    #[test]
    fn avoids_false_positive_on_one_div_40_100_020_seed30_runtime_path() {
        let mut base = SimulationConfig::default();
        base.n_steps = 10_000;
        let config = HyperparameterVariance::default().apply(&base, 30);

        let result = catch_unwind(AssertUnwindSafe(|| {
            engine::run_simulation_native(
                one_div_40_100_020_swap,
                None,
                normalizer_swap,
                None,
                &config,
            )
            .expect("simulation should complete");
        }));

        assert!(
            result.is_ok(),
            "one-div 40/100/0.020 strategy should not trip runtime shape checks"
        );
    }

    #[test]
    fn avoids_false_positive_on_one_div_30_80_019_seed72_runtime_path() {
        let mut base = SimulationConfig::default();
        base.n_steps = 10_000;
        let config = HyperparameterVariance::default().apply(&base, 72);

        let result = catch_unwind(AssertUnwindSafe(|| {
            engine::run_simulation_native(
                one_div_30_80_019_swap,
                None,
                normalizer_swap,
                None,
                &config,
            )
            .expect("simulation should complete");
        }));

        assert!(
            result.is_ok(),
            "one-div 30/80/0.019 strategy should not trip runtime shape checks"
        );
    }
}
