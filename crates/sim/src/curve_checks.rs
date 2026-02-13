use std::cmp::Ordering;

const X_REL_EPS: f64 = 1e-9;
const X_ABS_EPS: f64 = 1e-12;
const OUTPUT_REL_TOL: f64 = 1e-9;
const OUTPUT_ABS_TOL: f64 = 1e-9;
const SLOPE_REL_TOL: f64 = 1e-2;
const SLOPE_ABS_TOL: f64 = 1e-8;

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
    let mut sorted: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(input, output)| {
            input.is_finite() && output.is_finite() && *input > min_input && *output >= 0.0
        })
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut cleaned: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (input, output) in sorted {
        if let Some((prev_input, prev_output)) = cleaned.last_mut() {
            let eps = X_ABS_EPS.max(X_REL_EPS * prev_input.abs().max(input.abs()).max(1.0));
            if (input - *prev_input).abs() <= eps {
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
        let allowed_drop = OUTPUT_ABS_TOL + OUTPUT_REL_TOL * out_a.abs().max(out_b.abs()).max(1.0);
        if in_b > in_a && out_b + allowed_drop < out_a {
            return Some(format!(
                "monotonicity violated: input {in_a:.6} -> output {out_a:.6}, \
                 input {in_b:.6} -> output {out_b:.6}"
            ));
        }
    }

    let mut prev_slope: Option<f64> = None;
    for window in cleaned.windows(2) {
        let (in_a, out_a) = window[0];
        let (in_b, out_b) = window[1];
        let dx = in_b - in_a;
        if dx <= X_ABS_EPS {
            continue;
        }
        let slope = (out_b - out_a) / dx;
        if let Some(prev) = prev_slope {
            let scale = prev.abs().max(slope.abs()).max(1e-6);
            let allowed_rise = SLOPE_ABS_TOL + SLOPE_REL_TOL * scale;
            if slope > prev + allowed_rise {
                return Some(format!(
                    "concavity violated: slope rose from {prev:.9} to {slope:.9} \
                     between inputs {in_a:.6} and {in_b:.6}"
                ));
            }
        }
        prev_slope = Some(slope);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::submission_shape_violation;
    use crate::amm::BpfAmm;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    const MIN_INPUT: f64 = 1e-3;

    fn assert_valid(points: &[(f64, f64)], context: &str) {
        if let Some(err) = submission_shape_violation(points, MIN_INPUT) {
            panic!("{context}: unexpected shape violation: {err}");
        }
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
}
