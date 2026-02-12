use crate::amm::BpfAmm;
use prop_amm_shared::nano::{f64_to_nano, NANO_SCALE_F64};
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use rand_pcg::Pcg64;

const MIN_INPUT: f64 = 0.001;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_MAX_ITERS: usize = 24;
const GOLDEN_REL_TOL: f64 = 1e-6;
const BRACKET_MAX_STEPS: usize = 24;
const BRACKET_GROWTH: f64 = 2.0;
const MAX_INPUT_AMOUNT: f64 = (u64::MAX as f64 / NANO_SCALE_F64) * 0.999_999;

pub struct ArbResult {
    pub amm_buys_x: bool,
    pub amount_x: f64,
    pub amount_y: f64,
    pub edge: f64,
}

pub struct Arbitrageur {
    min_arb_profit: f64,
    rng: Pcg64,
    retail_size_dist: LogNormal<f64>,
}

impl Arbitrageur {
    pub fn new(
        min_arb_profit: f64,
        retail_mean_size: f64,
        retail_size_sigma: f64,
        seed: u64,
    ) -> Self {
        let sigma = retail_size_sigma.max(0.01);
        let mu_ln = retail_mean_size.max(0.01).ln() - 0.5 * sigma * sigma;
        Self {
            min_arb_profit: min_arb_profit.max(0.0),
            rng: Pcg64::seed_from_u64(seed),
            retail_size_dist: LogNormal::new(mu_ln, sigma).unwrap(),
        }
    }

    pub fn execute_arb(&mut self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let spot = amm.spot_price();
        if !spot.is_finite() || !fair_price.is_finite() || fair_price <= 0.0 {
            return None;
        }

        if spot < fair_price * 0.9999 {
            self.arb_buy_x(amm, fair_price)
        } else if spot > fair_price * 1.0001 {
            self.arb_sell_x(amm, fair_price)
        } else {
            None
        }
    }

    fn sample_retail_size_y(&mut self) -> f64 {
        self.retail_size_dist.sample(&mut self.rng).max(MIN_INPUT)
    }

    fn arb_buy_x(&mut self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let start_y = self.sample_retail_size_y().min(MAX_INPUT_AMOUNT);
        let mut sampled_curve = Vec::with_capacity(BRACKET_MAX_STEPS + GOLDEN_MAX_ITERS + 8);
        let (lo, hi) = Self::bracket_maximum(start_y, MAX_INPUT_AMOUNT, |input_y| {
            let output_x = amm.quote_buy_x(input_y);
            sampled_curve.push((input_y, output_x));
            output_x * fair_price - input_y
        });
        let (optimal_y, _) = Self::golden_section_max(lo, hi, |input_y| {
            let output_x = amm.quote_buy_x(input_y);
            sampled_curve.push((input_y, output_x));
            output_x * fair_price - input_y
        });
        Self::enforce_submission_curve_shape(amm, &sampled_curve, amm.reserve_x, "buy");

        if optimal_y < MIN_INPUT {
            return None;
        }

        let expected_output_x = amm.quote_buy_x(optimal_y);
        if expected_output_x <= 0.0 {
            return None;
        }

        let arb_profit = expected_output_x * fair_price - optimal_y;
        if arb_profit < self.min_arb_profit {
            return None;
        }

        let output_x = amm.execute_buy_x(optimal_y);
        if output_x <= 0.0 {
            return None;
        }

        Some(ArbResult {
            amm_buys_x: false,
            amount_x: output_x,
            amount_y: optimal_y,
            edge: optimal_y - output_x * fair_price,
        })
    }

    fn arb_sell_x(&mut self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let start_x = (self.sample_retail_size_y() / fair_price.max(1e-9))
            .max(MIN_INPUT)
            .min(MAX_INPUT_AMOUNT);
        let mut sampled_curve = Vec::with_capacity(BRACKET_MAX_STEPS + GOLDEN_MAX_ITERS + 8);
        let (lo, hi) = Self::bracket_maximum(start_x, MAX_INPUT_AMOUNT, |input_x| {
            let output_y = amm.quote_sell_x(input_x);
            sampled_curve.push((input_x, output_y));
            output_y - input_x * fair_price
        });
        let (optimal_x, _) = Self::golden_section_max(lo, hi, |input_x| {
            let output_y = amm.quote_sell_x(input_x);
            sampled_curve.push((input_x, output_y));
            output_y - input_x * fair_price
        });
        Self::enforce_submission_curve_shape(amm, &sampled_curve, amm.reserve_y, "sell");

        if optimal_x < MIN_INPUT {
            return None;
        }

        let expected_output_y = amm.quote_sell_x(optimal_x);
        if expected_output_y <= 0.0 {
            return None;
        }

        let arb_profit = expected_output_y - optimal_x * fair_price;
        if arb_profit < self.min_arb_profit {
            return None;
        }

        let output_y = amm.execute_sell_x(optimal_x);
        if output_y <= 0.0 {
            return None;
        }

        Some(ArbResult {
            amm_buys_x: true,
            amount_x: optimal_x,
            amount_y: output_y,
            edge: optimal_x * fair_price - output_y,
        })
    }

    fn bracket_maximum<F>(start: f64, max_input: f64, mut objective: F) -> (f64, f64)
    where
        F: FnMut(f64) -> f64,
    {
        let mut lo = 0.0_f64;
        let max_input = max_input.max(MIN_INPUT);
        let mut mid = start.clamp(MIN_INPUT, max_input);
        let mut mid_value = Self::sanitize_score(objective(mid));

        // Profit at zero input is always zero.
        if mid_value <= 0.0 {
            return (lo, mid);
        }

        let mut hi = (mid * BRACKET_GROWTH).min(max_input);
        if hi <= mid {
            return (lo, mid);
        }
        let mut hi_value = Self::sanitize_score(objective(hi));

        for _ in 0..BRACKET_MAX_STEPS {
            if hi_value <= mid_value || hi >= max_input {
                return (lo, hi);
            }

            lo = mid;
            mid = hi;
            mid_value = hi_value;

            let next_hi = (hi * BRACKET_GROWTH).min(max_input);
            if next_hi <= hi {
                return (lo, hi);
            }
            hi = next_hi;
            hi_value = Self::sanitize_score(objective(hi));
        }

        (lo, hi)
    }

    fn golden_section_max<F>(lo: f64, hi: f64, mut objective: F) -> (f64, f64)
    where
        F: FnMut(f64) -> f64,
    {
        let mut left = lo.min(hi).max(0.0);
        let mut right = hi.max(lo).max(MIN_INPUT);

        if right <= left {
            let value = Self::sanitize_score(objective(right));
            return (right, value);
        }

        let mut best_x = left;
        let mut best_value = Self::sanitize_score(objective(left));
        let right_value = Self::sanitize_score(objective(right));
        if right_value > best_value {
            best_x = right;
            best_value = right_value;
        }

        let mut x1 = right - GOLDEN_RATIO_CONJUGATE * (right - left);
        let mut x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
        let mut f1 = Self::sanitize_score(objective(x1));
        let mut f2 = Self::sanitize_score(objective(x2));
        if f1 > best_value {
            best_x = x1;
            best_value = f1;
        }
        if f2 > best_value {
            best_x = x2;
            best_value = f2;
        }

        for _ in 0..GOLDEN_MAX_ITERS {
            if f1 < f2 {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
                f2 = Self::sanitize_score(objective(x2));
                if f2 > best_value {
                    best_x = x2;
                    best_value = f2;
                }
            } else {
                right = x2;
                x2 = x1;
                f2 = f1;
                x1 = right - GOLDEN_RATIO_CONJUGATE * (right - left);
                f1 = Self::sanitize_score(objective(x1));
                if f1 > best_value {
                    best_x = x1;
                    best_value = f1;
                }
            }

            if (right - left) <= GOLDEN_REL_TOL * (1.0 + left + right) {
                break;
            }
        }

        let center = 0.5 * (left + right);
        let center_value = Self::sanitize_score(objective(center));
        if center_value > best_value {
            (center, center_value)
        } else {
            (best_x, best_value)
        }
    }

    #[inline]
    fn sanitize_score(value: f64) -> f64 {
        if value.is_finite() {
            value
        } else {
            f64::NEG_INFINITY
        }
    }

    fn enforce_submission_curve_shape(
        amm: &BpfAmm,
        points: &[(f64, f64)],
        max_output: f64,
        side_label: &str,
    ) {
        if amm.name != "submission" || !amm.uses_bpf_backend() {
            return;
        }
        if !Self::is_curve_shape_valid(points, max_output) {
            panic!(
                "submission curve shape violation detected during arbitrage {} search",
                side_label
            );
        }
    }

    fn is_curve_shape_valid(points: &[(f64, f64)], max_output: f64) -> bool {
        const MIN_INPUT_NANO: u64 = 1_000_000; // 0.001 units

        let max_output_nano = f64_to_nano(max_output);
        if max_output_nano == 0 {
            return false;
        }
        if points.is_empty() {
            return true;
        }

        // Validate in nano-space to avoid floating-point artifacts.
        let mut sorted: Vec<(u64, u64)> = points
            .iter()
            .filter_map(|(input, output)| {
                if !input.is_finite() || !output.is_finite() || *input < 0.0 {
                    return None;
                }
                let input_nano = f64_to_nano(*input);
                let output_nano = f64_to_nano(*output);
                if input_nano < MIN_INPUT_NANO
                    || output_nano == 0
                    || output_nano >= max_output_nano
                {
                    return None;
                }
                Some((input_nano, output_nano))
            })
            .collect();
        if sorted.len() < 3 {
            return true;
        }
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // Collapse duplicate inputs to a single best-observed output.
        let mut collapsed: Vec<(u64, u64)> = Vec::with_capacity(sorted.len());
        for (input_nano, output_nano) in sorted {
            if let Some((last_in, last_out)) = collapsed.last_mut() {
                if *last_in == input_nano {
                    *last_out = (*last_out).max(output_nano);
                    continue;
                }
            }
            collapsed.push((input_nano, output_nano));
        }

        if collapsed.len() < 3 {
            return true;
        }

        const MIN_DIN: u64 = 100_000;

        let mut landmarks: Vec<(u64, u64)> = Vec::new();
        for &(in_n, out_n) in &collapsed {
            if let Some(&(last_in, _)) = landmarks.last() {
                if in_n - last_in < MIN_DIN {
                    continue;
                }
            }
            landmarks.push((in_n, out_n));
        }

        if landmarks.len() < 3 {
            return true;
        }

        let mut prev_slope = f64::INFINITY;
        for window in landmarks.windows(2) {
            let (in_a, out_a) = window[0];
            let (in_b, out_b) = window[1];
            if out_b + 1 < out_a {
                return false;
            }
            let din = (in_b - in_a) as f64;
            let dout = out_b.saturating_sub(out_a) as f64;
            let slope = dout / din;
            if slope < 0.0 {
                return false;
            }
            let ref_slope = if prev_slope.is_finite() {
                prev_slope.max(slope)
            } else {
                slope
            };
            let slope_rounding_tol = ref_slope * 5e-4 + 8.0 / din;
            if slope > prev_slope + slope_rounding_tol {
                return false;
            }
            prev_slope = slope;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::Arbitrageur;
    use crate::amm::BpfAmm;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;

    fn test_amm() -> BpfAmm {
        BpfAmm::new_native(normalizer_swap, None, 100.0, 10_000.0, "test".to_string())
    }

    #[test]
    fn min_arb_profit_blocks_profitable_trade_when_threshold_is_higher() {
        let fair_price = 101.0;

        let mut amm_without_floor = test_amm();
        let mut no_floor = Arbitrageur::new(0.0, 20.0, 1.2, 42);
        let result = no_floor
            .execute_arb(&mut amm_without_floor, fair_price)
            .expect("expected profitable arbitrage");
        let realized_profit = -result.edge;
        assert!(
            realized_profit > 0.0,
            "arb should produce positive arb profit"
        );

        let mut amm_with_floor = test_amm();
        let mut floor = Arbitrageur::new(realized_profit + 1e-9, 20.0, 1.2, 42);
        assert!(
            floor.execute_arb(&mut amm_with_floor, fair_price).is_none(),
            "trade should be skipped when profit ({realized_profit}) is below threshold"
        );
    }
}
