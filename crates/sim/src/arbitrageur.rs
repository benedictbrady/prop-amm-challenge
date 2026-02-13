use crate::amm::BpfAmm;
use crate::search_stats;
use prop_amm_shared::nano::NANO_SCALE_F64;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use rand_pcg::Pcg64;

const MIN_INPUT: f64 = 0.001;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_MAX_ITERS: usize = 12;
// Stop once the bracket is narrow enough that the trade size is within ~5%.
const GOLDEN_INPUT_REL_TOL: f64 = 5e-2;
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

        // The normalizer is a known constant-product-with-fee curve. Arb it with a closed-form
        // solution to avoid dozens of quote calls per step.
        if amm.name == "normalizer" {
            return self.execute_normalizer_closed_form(amm, fair_price, spot);
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

    fn execute_normalizer_closed_form(
        &mut self,
        amm: &mut BpfAmm,
        fair_price: f64,
        spot: f64,
    ) -> Option<ArbResult> {
        debug_assert_eq!(amm.name, "normalizer");

        let fee_bps = Self::normalizer_fee_bps(amm) as f64;
        let gamma = (10_000.0 - fee_bps) / 10_000.0;
        if !gamma.is_finite() || gamma <= 0.0 {
            return None;
        }

        let rx = amm.reserve_x;
        let ry = amm.reserve_y;
        if !rx.is_finite() || !ry.is_finite() || rx <= 0.0 || ry <= 0.0 {
            return None;
        }

        if spot < fair_price * 0.9999 {
            // Buy X with Y
            let target = (fair_price * rx * gamma * ry).sqrt();
            if !target.is_finite() || target <= ry {
                return None;
            }
            let input_y = ((target - ry) / gamma).clamp(MIN_INPUT, MAX_INPUT_AMOUNT);
            let expected_output_x = amm.quote_buy_x(input_y);
            if expected_output_x <= 0.0 {
                return None;
            }
            let arb_profit = expected_output_x * fair_price - input_y;
            if arb_profit < self.min_arb_profit {
                return None;
            }
            let output_x = amm.execute_buy_x(input_y);
            if output_x <= 0.0 {
                return None;
            }
            Some(ArbResult {
                amm_buys_x: false,
                amount_x: output_x,
                amount_y: input_y,
                edge: input_y - output_x * fair_price,
            })
        } else if spot > fair_price * 1.0001 {
            // Sell X for Y
            let target = (ry * rx * gamma / fair_price).sqrt();
            if !target.is_finite() || target <= rx {
                return None;
            }
            let input_x = ((target - rx) / gamma).clamp(MIN_INPUT, MAX_INPUT_AMOUNT);
            let expected_output_y = amm.quote_sell_x(input_x);
            if expected_output_y <= 0.0 {
                return None;
            }
            let arb_profit = expected_output_y - input_x * fair_price;
            if arb_profit < self.min_arb_profit {
                return None;
            }
            let output_y = amm.execute_sell_x(input_x);
            if output_y <= 0.0 {
                return None;
            }
            Some(ArbResult {
                amm_buys_x: true,
                amount_x: input_x,
                amount_y: output_y,
                edge: input_x * fair_price - output_y,
            })
        } else {
            None
        }
    }

    #[inline]
    fn normalizer_fee_bps(amm: &BpfAmm) -> u16 {
        // normalizer::compute_swap reads fee_bps from data[25..27], i.e. storage[0..2].
        let s = amm.storage();
        if s.len() >= 2 {
            let raw = u16::from_le_bytes([s[0], s[1]]);
            if raw == 0 { 30 } else { raw }
        } else {
            30
        }
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
        Self::enforce_submission_monotonicity(amm, &sampled_curve, "buy");

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
        Self::enforce_submission_monotonicity(amm, &sampled_curve, "sell");

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
        search_stats::inc_arb_bracket_call();
        let mut lo = 0.0_f64;
        let max_input = max_input.max(MIN_INPUT);
        let mut mid = start.clamp(MIN_INPUT, max_input);
        search_stats::inc_arb_bracket_eval();
        let mut mid_value = Self::sanitize_score(objective(mid));

        // Profit at zero input is always zero.
        if mid_value <= 0.0 {
            return (lo, mid);
        }

        let mut hi = (mid * BRACKET_GROWTH).min(max_input);
        if hi <= mid {
            return (lo, mid);
        }
        search_stats::inc_arb_bracket_eval();
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
            search_stats::inc_arb_bracket_eval();
            hi_value = Self::sanitize_score(objective(hi));
        }

        (lo, hi)
    }

    fn golden_section_max<F>(lo: f64, hi: f64, mut objective: F) -> (f64, f64)
    where
        F: FnMut(f64) -> f64,
    {
        search_stats::inc_arb_golden_call();
        let mut left = lo.min(hi).max(0.0);
        let mut right = hi.max(lo).max(MIN_INPUT);

        if right <= left {
            search_stats::inc_arb_golden_eval();
            let value = Self::sanitize_score(objective(right));
            return (right, value);
        }

        let mut best_x = left;
        search_stats::inc_arb_golden_eval();
        let mut best_value = Self::sanitize_score(objective(left));
        search_stats::inc_arb_golden_eval();
        let right_value = Self::sanitize_score(objective(right));
        if right_value > best_value {
            best_x = right;
            best_value = right_value;
        }

        let mut x1 = right - GOLDEN_RATIO_CONJUGATE * (right - left);
        let mut x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
        search_stats::inc_arb_golden_eval();
        let mut f1 = Self::sanitize_score(objective(x1));
        search_stats::inc_arb_golden_eval();
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
            search_stats::inc_arb_golden_iter();
            if f1 < f2 {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
                search_stats::inc_arb_golden_eval();
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
                search_stats::inc_arb_golden_eval();
                f1 = Self::sanitize_score(objective(x1));
                if f1 > best_value {
                    best_x = x1;
                    best_value = f1;
                }
            }

            // Use bracket width in x-space as the stopping condition: we care about sizing
            // the trade, not precisely maximizing profit.
            let mid = 0.5 * (left + right);
            let denom = mid.abs().max(MIN_INPUT);
            if (right - left) <= GOLDEN_INPUT_REL_TOL * denom {
                search_stats::inc_arb_early_stop_amount_tol();
                break;
            }
        }

        // With loose tolerances, a final center evaluation is rarely worth another quote call.
        (best_x, best_value)
    }

    #[inline]
    fn sanitize_score(value: f64) -> f64 {
        if value.is_finite() {
            value
        } else {
            f64::NEG_INFINITY
        }
    }

    fn enforce_submission_monotonicity(
        amm: &BpfAmm,
        points: &[(f64, f64)],
        side_label: &str,
    ) {
        if amm.name != "submission" {
            return;
        }
        let mut sorted: Vec<(f64, f64)> = points
            .iter()
            .copied()
            .filter(|(i, o)| i.is_finite() && o.is_finite() && *i > MIN_INPUT)
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for window in sorted.windows(2) {
            let (in_a, out_a) = window[0];
            let (in_b, out_b) = window[1];
            if in_b > in_a + 1e-9 && out_b + 1e-9 < out_a {
                panic!(
                    "submission monotonicity violation during arbitrage {side_label} search: \
                     input {in_a:.6} -> output {out_a:.6}, input {in_b:.6} -> output {out_b:.6}"
                );
            }
        }
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
