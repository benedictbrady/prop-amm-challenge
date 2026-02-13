use crate::amm::BpfAmm;
use crate::curve_checks;
use crate::search_stats;
use prop_amm_shared::nano::NANO_SCALE_F64;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use rand_pcg::Pcg64;

const MIN_INPUT: f64 = 0.001;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_MAX_ITERS: usize = 12;
// Stop once the bracket is narrow enough that the trade size is within ~1%.
const GOLDEN_INPUT_REL_TOL: f64 = 1e-2;
const BRACKET_MAX_STEPS: usize = 24;
const BRACKET_GROWTH: f64 = 2.0;
const MAX_INPUT_AMOUNT: f64 = (u64::MAX as f64 / NANO_SCALE_F64) * 0.999_999;
// Ignore micro-arbs by requiring a minimum quote-token (Y) notional.
const MIN_ARB_NOTIONAL_Y: f64 = 0.01;

#[derive(Clone, Copy)]
enum ArbSide {
    BuyX,
    SellX,
}

#[derive(Clone, Copy)]
struct ArbCandidate {
    side: ArbSide,
    input_amount: f64,
    expected_profit: f64,
}

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
        if !fair_price.is_finite() || fair_price <= 0.0 {
            return None;
        }

        let best = if amm.name == "normalizer" {
            // The normalizer is a known constant-product-with-fee curve. Keep it closed-form,
            // but evaluate both sides and execute whichever quote-implied trade is better.
            Self::best_candidate(
                self.plan_normalizer_buy_x(amm, fair_price),
                self.plan_normalizer_sell_x(amm, fair_price),
            )
        } else {
            // Evaluate both book sides from compute_swap quotes; reserve_y/reserve_x can be a
            // misleading directional signal for non-CP strategies.
            let min_buy_input = Self::min_buy_input_y();
            let min_sell_input = Self::min_sell_input_x(fair_price);
            let start_y = self
                .sample_retail_size_y()
                .max(min_buy_input)
                .min(MAX_INPUT_AMOUNT);
            let start_x = (start_y / fair_price.max(1e-9))
                .max(min_sell_input)
                .min(MAX_INPUT_AMOUNT);
            Self::best_candidate(
                self.plan_arb_buy_x(amm, fair_price, start_y, min_buy_input),
                self.plan_arb_sell_x(amm, fair_price, start_x, min_sell_input),
            )
        }?;

        self.execute_candidate(amm, fair_price, best)
    }

    fn sample_retail_size_y(&mut self) -> f64 {
        self.retail_size_dist.sample(&mut self.rng).max(MIN_INPUT)
    }

    #[inline]
    fn min_buy_input_y() -> f64 {
        MIN_INPUT.max(MIN_ARB_NOTIONAL_Y)
    }

    #[inline]
    fn min_sell_input_x(fair_price: f64) -> f64 {
        MIN_INPUT.max(MIN_ARB_NOTIONAL_Y / fair_price.max(1e-9))
    }

    fn plan_normalizer_buy_x(&self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbCandidate> {
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

        // Buy X with Y
        let target = (fair_price * rx * gamma * ry).sqrt();
        if !target.is_finite() || target <= ry {
            None
        } else {
            let min_buy_input = Self::min_buy_input_y();
            let input_y = ((target - ry) / gamma).clamp(min_buy_input, MAX_INPUT_AMOUNT);
            let expected_output_x = amm.quote_buy_x(input_y);
            if expected_output_x <= 0.0 {
                return None;
            }
            let arb_profit = expected_output_x * fair_price - input_y;
            if arb_profit < self.min_arb_profit {
                return None;
            }
            Some(ArbCandidate {
                side: ArbSide::BuyX,
                input_amount: input_y,
                expected_profit: arb_profit,
            })
        }
    }

    fn plan_normalizer_sell_x(&self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbCandidate> {
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

        // Sell X for Y
        let target = (ry * rx * gamma / fair_price).sqrt();
        if !target.is_finite() || target <= rx {
            None
        } else {
            let min_sell_input = Self::min_sell_input_x(fair_price);
            let input_x = ((target - rx) / gamma).clamp(min_sell_input, MAX_INPUT_AMOUNT);
            let expected_output_y = amm.quote_sell_x(input_x);
            if expected_output_y <= 0.0 {
                return None;
            }
            let arb_profit = expected_output_y - input_x * fair_price;
            if arb_profit < self.min_arb_profit {
                return None;
            }
            Some(ArbCandidate {
                side: ArbSide::SellX,
                input_amount: input_x,
                expected_profit: arb_profit,
            })
        }
    }

    #[inline]
    fn normalizer_fee_bps(amm: &BpfAmm) -> u16 {
        // normalizer::compute_swap reads fee_bps from data[25..27], i.e. storage[0..2].
        let s = amm.storage();
        if s.len() >= 2 {
            let raw = u16::from_le_bytes([s[0], s[1]]);
            if raw == 0 {
                30
            } else {
                raw
            }
        } else {
            30
        }
    }

    fn plan_arb_buy_x(
        &mut self,
        amm: &mut BpfAmm,
        fair_price: f64,
        start_y: f64,
        min_buy_input: f64,
    ) -> Option<ArbCandidate> {
        let mut sampled_curve = Vec::with_capacity(BRACKET_MAX_STEPS + GOLDEN_MAX_ITERS + 8);
        let (lo, hi) = Self::bracket_maximum(start_y, min_buy_input, MAX_INPUT_AMOUNT, |input_y| {
            let output_x = amm.quote_buy_x(input_y);
            sampled_curve.push((input_y, output_x));
            output_x * fair_price - input_y
        });
        let (optimal_y, _) = Self::golden_section_max(lo, hi, |input_y| {
            let output_x = amm.quote_buy_x(input_y);
            sampled_curve.push((input_y, output_x));
            output_x * fair_price - input_y
        });
        curve_checks::enforce_submission_monotonic_concave(
            &amm.name,
            &sampled_curve,
            min_buy_input,
            "arbitrage buy search",
        );

        if optimal_y < min_buy_input {
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

        Some(ArbCandidate {
            side: ArbSide::BuyX,
            input_amount: optimal_y,
            expected_profit: arb_profit,
        })
    }

    fn plan_arb_sell_x(
        &mut self,
        amm: &mut BpfAmm,
        fair_price: f64,
        start_x: f64,
        min_sell_input: f64,
    ) -> Option<ArbCandidate> {
        let mut sampled_curve = Vec::with_capacity(BRACKET_MAX_STEPS + GOLDEN_MAX_ITERS + 8);
        let (lo, hi) =
            Self::bracket_maximum(start_x, min_sell_input, MAX_INPUT_AMOUNT, |input_x| {
            let output_y = amm.quote_sell_x(input_x);
            sampled_curve.push((input_x, output_y));
            output_y - input_x * fair_price
        });
        let (optimal_x, _) = Self::golden_section_max(lo, hi, |input_x| {
            let output_y = amm.quote_sell_x(input_x);
            sampled_curve.push((input_x, output_y));
            output_y - input_x * fair_price
        });
        curve_checks::enforce_submission_monotonic_concave(
            &amm.name,
            &sampled_curve,
            min_sell_input,
            "arbitrage sell search",
        );

        if optimal_x < min_sell_input {
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

        Some(ArbCandidate {
            side: ArbSide::SellX,
            input_amount: optimal_x,
            expected_profit: arb_profit,
        })
    }

    fn best_candidate(
        buy: Option<ArbCandidate>,
        sell: Option<ArbCandidate>,
    ) -> Option<ArbCandidate> {
        match (buy, sell) {
            (Some(buy), Some(sell)) => {
                if sell.expected_profit > buy.expected_profit {
                    Some(sell)
                } else {
                    Some(buy)
                }
            }
            (Some(buy), None) => Some(buy),
            (None, Some(sell)) => Some(sell),
            (None, None) => None,
        }
    }

    fn execute_candidate(
        &self,
        amm: &mut BpfAmm,
        fair_price: f64,
        candidate: ArbCandidate,
    ) -> Option<ArbResult> {
        match candidate.side {
            ArbSide::BuyX => {
                let output_x = amm.execute_buy_x(candidate.input_amount);
                if output_x <= 0.0 {
                    return None;
                }
                Some(ArbResult {
                    amm_buys_x: false,
                    amount_x: output_x,
                    amount_y: candidate.input_amount,
                    edge: candidate.input_amount - output_x * fair_price,
                })
            }
            ArbSide::SellX => {
                let output_y = amm.execute_sell_x(candidate.input_amount);
                if output_y <= 0.0 {
                    return None;
                }
                Some(ArbResult {
                    amm_buys_x: true,
                    amount_x: candidate.input_amount,
                    amount_y: output_y,
                    edge: candidate.input_amount * fair_price - output_y,
                })
            }
        }
    }

    fn bracket_maximum<F>(
        start: f64,
        min_input: f64,
        max_input: f64,
        mut objective: F,
    ) -> (f64, f64)
    where
        F: FnMut(f64) -> f64,
    {
        search_stats::inc_arb_bracket_call();
        let min_input = min_input.max(MIN_INPUT);
        let mut lo = min_input;
        let max_input = max_input.max(min_input);
        let mut mid = start.clamp(min_input, max_input);
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
}

#[cfg(test)]
mod tests {
    use super::Arbitrageur;
    use crate::amm::BpfAmm;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;

    const NANO_SCALE: f64 = 1_000_000_000.0;

    fn test_amm() -> BpfAmm {
        BpfAmm::new_native(normalizer_swap, None, 100.0, 10_000.0, "test".to_string())
    }

    fn to_nano_u64(amount: f64) -> u64 {
        if !amount.is_finite() || amount <= 0.0 {
            return 0;
        }
        let scaled = (amount * NANO_SCALE).floor();
        if scaled <= 0.0 {
            0
        } else if scaled >= u64::MAX as f64 {
            u64::MAX
        } else {
            scaled as u64
        }
    }

    fn linear_quote_swap(data: &[u8], buy_price: f64, sell_price: f64) -> u64 {
        if data.len() < 25 {
            return 0;
        }
        let side = data[0];
        let input = u64::from_le_bytes(data[1..9].try_into().expect("input")) as f64 / NANO_SCALE;
        if !input.is_finite() || input <= 0.0 {
            return 0;
        }
        let output = match side {
            0 => input / buy_price,
            1 => input * sell_price,
            _ => 0.0,
        };
        to_nano_u64(output)
    }

    fn fixed_price_120_swap(data: &[u8]) -> u64 {
        linear_quote_swap(data, 120.0, 120.0)
    }

    fn crossed_price_swap(data: &[u8]) -> u64 {
        linear_quote_swap(data, 99.0, 120.0)
    }

    fn subfloor_buy_only_swap(data: &[u8]) -> u64 {
        if data.len() < 25 {
            return 0;
        }
        let side = data[0];
        let input = u64::from_le_bytes(data[1..9].try_into().expect("input")) as f64 / NANO_SCALE;
        if !input.is_finite() || input <= 0.0 {
            return 0;
        }
        let output = match side {
            // Profitable for tiny buys (< 0.01 Y), unprofitable at/above the floor.
            0 => {
                if input < 0.01 {
                    input / 10.0
                } else {
                    input / 120.0
                }
            }
            // Never profitable on sell side.
            1 => 0.0,
            _ => 0.0,
        };
        to_nano_u64(output)
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

    #[test]
    fn explores_opposite_side_when_reserve_spot_direction_is_wrong() {
        let fair_price = 100.5;
        let mut amm = BpfAmm::new_native(
            fixed_price_120_swap,
            None,
            100.0,
            10_000.0,
            "test".to_string(),
        );
        assert!(
            amm.spot_price() < fair_price * 0.9999,
            "reserve spot should suggest buy-X under legacy gating"
        );

        let buy_probe_profit = amm.quote_buy_x(20.0) * fair_price - 20.0;
        let sell_probe_profit = amm.quote_sell_x(0.2) - 0.2 * fair_price;
        assert!(buy_probe_profit < 0.0, "buy side should be unprofitable");
        assert!(sell_probe_profit > 0.0, "sell side should be profitable");

        let mut arb = Arbitrageur::new(0.01, 20.0, 1.2, 7);
        let result = arb
            .execute_arb(&mut amm, fair_price)
            .expect("arb should execute profitable sell-X trade");
        assert!(result.amm_buys_x, "trade should be sell-X (AMM buys X)");
    }

    #[test]
    fn chooses_higher_profit_side_when_both_are_profitable() {
        let fair_price = 100.5;
        let mut amm = BpfAmm::new_native(
            crossed_price_swap,
            None,
            100.0,
            10_000.0,
            "test".to_string(),
        );
        let buy_probe_profit = amm.quote_buy_x(20.0) * fair_price - 20.0;
        let sell_probe_profit = amm.quote_sell_x(0.2) - 0.2 * fair_price;
        assert!(buy_probe_profit > 0.0, "buy side should be profitable");
        assert!(sell_probe_profit > 0.0, "sell side should be profitable");
        assert!(
            sell_probe_profit > buy_probe_profit,
            "sell side should be clearly more profitable"
        );

        let mut arb = Arbitrageur::new(0.01, 20.0, 1.2, 17);
        let result = arb
            .execute_arb(&mut amm, fair_price)
            .expect("arb should execute one of the profitable trades");
        assert!(
            result.amm_buys_x,
            "arb should choose sell-X side with higher expected profit"
        );
    }

    #[test]
    fn ignores_profitable_subfloor_buy_arbs() {
        let fair_price = 100.0;
        let mut amm = BpfAmm::new_native(
            subfloor_buy_only_swap,
            None,
            100.0,
            10_000.0,
            "test".to_string(),
        );
        let small_profit = amm.quote_buy_x(0.005) * fair_price - 0.005;
        let floor_profit = amm.quote_buy_x(0.01) * fair_price - 0.01;
        assert!(
            small_profit > 0.01,
            "sub-floor trade should be profitable above arb threshold"
        );
        assert!(floor_profit <= 0.0, "at-floor trade should be unprofitable");

        let mut arb = Arbitrageur::new(0.01, 20.0, 1.2, 1234);
        assert!(
            arb.execute_arb(&mut amm, fair_price).is_none(),
            "arb should ignore opportunities below 0.01 Y notional floor"
        );
    }
}
