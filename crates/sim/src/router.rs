use crate::amm::BpfAmm;
use crate::curve_checks;
use crate::retail::RetailOrder;
use crate::search_stats;

pub struct RoutedTrade {
    pub is_submission: bool,
    pub amm_buys_x: bool,
    pub amount_x: f64,
    pub amount_y: f64,
}

const MIN_TRADE_SIZE: f64 = 0.001;
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_8;
const GOLDEN_MAX_ITERS: usize = 14;
const GOLDEN_ALPHA_TOL: f64 = 1e-3;
// Stop once the submission split amount is within ~1% (relative bracket width in amount-space).
const GOLDEN_SUBMISSION_AMOUNT_REL_TOL: f64 = 1e-2;
// Stop once the two evaluated total outputs are within 1% of each other.
const GOLDEN_SCORE_REL_GAP_TOL: f64 = 1e-2;

pub struct OrderRouter;

#[derive(Clone, Copy)]
struct QuotePoint {
    in_sub: f64,
    in_norm: f64,
    out_sub: f64,
    out_norm: f64,
}

struct SplitSearchResult {
    best: QuotePoint,
    sampled: Vec<QuotePoint>,
}

impl OrderRouter {
    pub fn new() -> Self {
        Self
    }

    pub fn route_order(
        &self,
        order: &RetailOrder,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
        fair_price: f64,
    ) -> Vec<RoutedTrade> {
        if order.is_buy {
            self.route_buy(order.size, amm_sub, amm_norm)
        } else {
            let total_x = order.size / fair_price;
            self.route_sell(total_x, amm_sub, amm_norm)
        }
    }

    fn route_buy(
        &self,
        total_y: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
    ) -> Vec<RoutedTrade> {
        let search = Self::maximize_split(total_y, |alpha| {
            Self::quote_buy_split(total_y, alpha, amm_sub, amm_norm)
        });
        curve_checks::enforce_submission_monotonic_concave(
            &amm_sub.name,
            &search
                .sampled
                .iter()
                .map(|p| (p.in_sub, p.out_sub))
                .collect::<Vec<_>>(),
            MIN_TRADE_SIZE,
            "router buy split search",
        );
        let best = search.best;

        let mut trades = Vec::new();
        let y_sub = best.in_sub;
        let y_norm = best.in_norm;

        if y_sub > MIN_TRADE_SIZE && best.out_sub > 0.0 {
            let x_out = amm_sub.execute_buy_x(y_sub);
            if x_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: true,
                    amm_buys_x: false,
                    amount_x: x_out,
                    amount_y: y_sub,
                });
            }
        }
        if y_norm > MIN_TRADE_SIZE && best.out_norm > 0.0 {
            let x_out = amm_norm.execute_buy_x(y_norm);
            if x_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: false,
                    amm_buys_x: false,
                    amount_x: x_out,
                    amount_y: y_norm,
                });
            }
        }
        trades
    }

    fn route_sell(
        &self,
        total_x: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
    ) -> Vec<RoutedTrade> {
        let search = Self::maximize_split(total_x, |alpha| {
            Self::quote_sell_split(total_x, alpha, amm_sub, amm_norm)
        });
        curve_checks::enforce_submission_monotonic_concave(
            &amm_sub.name,
            &search
                .sampled
                .iter()
                .map(|p| (p.in_sub, p.out_sub))
                .collect::<Vec<_>>(),
            MIN_TRADE_SIZE,
            "router sell split search",
        );
        let best = search.best;

        let mut trades = Vec::new();
        let x_sub = best.in_sub;
        let x_norm = best.in_norm;

        if x_sub > MIN_TRADE_SIZE && best.out_sub > 0.0 {
            let y_out = amm_sub.execute_sell_x(x_sub);
            if y_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: true,
                    amm_buys_x: true,
                    amount_x: x_sub,
                    amount_y: y_out,
                });
            }
        }
        if x_norm > MIN_TRADE_SIZE && best.out_norm > 0.0 {
            let y_out = amm_norm.execute_sell_x(x_norm);
            if y_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: false,
                    amm_buys_x: true,
                    amount_x: x_norm,
                    amount_y: y_out,
                });
            }
        }
        trades
    }

    fn quote_buy_split(
        total_y: f64,
        alpha: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
    ) -> QuotePoint {
        let alpha = alpha.clamp(0.0, 1.0);
        let in_sub = total_y * alpha;
        let in_norm = total_y * (1.0 - alpha);

        let out_sub = if in_sub > MIN_TRADE_SIZE {
            amm_sub.quote_buy_x(in_sub)
        } else {
            0.0
        };
        let out_norm = if in_norm > MIN_TRADE_SIZE {
            amm_norm.quote_buy_x(in_norm)
        } else {
            0.0
        };

        QuotePoint {
            in_sub,
            in_norm,
            out_sub,
            out_norm,
        }
    }

    fn quote_sell_split(
        total_x: f64,
        alpha: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
    ) -> QuotePoint {
        let alpha = alpha.clamp(0.0, 1.0);
        let in_sub = total_x * alpha;
        let in_norm = total_x * (1.0 - alpha);

        let out_sub = if in_sub > MIN_TRADE_SIZE {
            amm_sub.quote_sell_x(in_sub)
        } else {
            0.0
        };
        let out_norm = if in_norm > MIN_TRADE_SIZE {
            amm_norm.quote_sell_x(in_norm)
        } else {
            0.0
        };

        QuotePoint {
            in_sub,
            in_norm,
            out_sub,
            out_norm,
        }
    }

    fn maximize_split<F>(total_input: f64, mut evaluate: F) -> SplitSearchResult
    where
        F: FnMut(f64) -> QuotePoint,
    {
        search_stats::inc_router_call();
        let mut sampled = Vec::with_capacity(GOLDEN_MAX_ITERS + 6);
        let mut left = 0.0_f64;
        let mut right = 1.0_f64;

        search_stats::inc_router_eval();
        let edge_left = evaluate(left);
        search_stats::inc_router_eval();
        let edge_right = evaluate(right);
        sampled.push(edge_left);
        sampled.push(edge_right);
        let mut best = Self::best_quote(edge_left, edge_right);

        let mut x1 = right - GOLDEN_RATIO_CONJUGATE * (right - left);
        let mut x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
        search_stats::inc_router_eval();
        let mut q1 = evaluate(x1);
        search_stats::inc_router_eval();
        let mut q2 = evaluate(x2);
        sampled.push(q1);
        sampled.push(q2);
        best = Self::best_quote(best, q1);
        best = Self::best_quote(best, q2);

        for _ in 0..GOLDEN_MAX_ITERS {
            search_stats::inc_router_iter();
            if right - left <= GOLDEN_ALPHA_TOL {
                break;
            }

            let alpha_mid = 0.5 * (left + right);
            let sub_mid_amount = total_input * alpha_mid;
            let amount_width = total_input * (right - left);
            let amount_scale = sub_mid_amount.abs().max(MIN_TRADE_SIZE);
            if amount_width <= GOLDEN_SUBMISSION_AMOUNT_REL_TOL * amount_scale {
                break;
            }

            if Self::within_rel_gap(
                Self::quote_score(&q1),
                Self::quote_score(&q2),
                GOLDEN_SCORE_REL_GAP_TOL,
            ) {
                search_stats::inc_router_early_stop_rel_gap();
                break;
            }

            if Self::quote_score(&q1) < Self::quote_score(&q2) {
                left = x1;
                x1 = x2;
                q1 = q2;
                x2 = left + GOLDEN_RATIO_CONJUGATE * (right - left);
                search_stats::inc_router_eval();
                q2 = evaluate(x2);
                sampled.push(q2);
                best = Self::best_quote(best, q2);
            } else {
                right = x2;
                x2 = x1;
                q2 = q1;
                x1 = right - GOLDEN_RATIO_CONJUGATE * (right - left);
                search_stats::inc_router_eval();
                q1 = evaluate(x1);
                sampled.push(q1);
                best = Self::best_quote(best, q1);
            }
        }

        search_stats::inc_router_eval();
        let center = evaluate((left + right) * 0.5);
        sampled.push(center);
        best = Self::best_quote(best, center);

        SplitSearchResult { best, sampled }
    }

    #[inline]
    fn quote_score(point: &QuotePoint) -> f64 {
        let total = point.out_sub + point.out_norm;
        if total.is_finite() {
            total
        } else {
            f64::NEG_INFINITY
        }
    }

    #[inline]
    fn within_rel_gap(a: f64, b: f64, rel_tol: f64) -> bool {
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        let denom = a.abs().max(b.abs()).max(1e-12);
        (a - b).abs() <= rel_tol * denom
    }

    #[inline]
    fn best_quote(a: QuotePoint, b: QuotePoint) -> QuotePoint {
        if Self::quote_score(&b) > Self::quote_score(&a) {
            b
        } else {
            a
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OrderRouter, MIN_TRADE_SIZE};
    use crate::amm::BpfAmm;
    use crate::retail::RetailOrder;
    use prop_amm_executor::SwapFn;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    const BRUTE_FORCE_STEPS: usize = 4000;
    // Router search is intentionally approximate for speed; 1% relative error is acceptable.
    const DIVERSE_CURVE_TOLERANCE: f64 = 1.0e-2;
    const ENDPOINT_REGIME_TOLERANCE: f64 = 1.0e-2;

    fn cp_fee_swap(data: &[u8], fee_numerator: u128, fee_denominator: u128) -> u64 {
        if data.len() < 25 {
            return 0;
        }

        let side = data[0];
        let input = u64::from_le_bytes(data[1..9].try_into().unwrap()) as u128;
        let rx = u64::from_le_bytes(data[9..17].try_into().unwrap()) as u128;
        let ry = u64::from_le_bytes(data[17..25].try_into().unwrap()) as u128;
        if rx == 0 || ry == 0 {
            return 0;
        }

        let k = rx.saturating_mul(ry);
        if k == 0 {
            return 0;
        }

        match side {
            0 => {
                let net = input.saturating_mul(fee_numerator) / fee_denominator;
                let new_ry = ry + net;
                rx.saturating_sub((k + new_ry - 1) / new_ry) as u64
            }
            1 => {
                let net = input.saturating_mul(fee_numerator) / fee_denominator;
                let new_rx = rx + net;
                ry.saturating_sub((k + new_rx - 1) / new_rx) as u64
            }
            _ => 0,
        }
    }

    fn zero_fee_swap(data: &[u8]) -> u64 {
        cp_fee_swap(data, 1_000, 1_000)
    }

    fn low_fee_swap(data: &[u8]) -> u64 {
        cp_fee_swap(data, 999, 1_000)
    }

    fn starter_fee_swap(data: &[u8]) -> u64 {
        cp_fee_swap(data, 995, 1_000)
    }

    fn high_fee_swap(data: &[u8]) -> u64 {
        cp_fee_swap(data, 50, 1_000)
    }

    fn total_output_from_trades(order: &RetailOrder, trades: &[crate::router::RoutedTrade]) -> f64 {
        if order.is_buy {
            trades.iter().map(|t| t.amount_x).sum()
        } else {
            trades.iter().map(|t| t.amount_y).sum()
        }
    }

    fn quote_total_output(
        order: &RetailOrder,
        fair_price: f64,
        alpha: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
    ) -> f64 {
        let alpha = alpha.clamp(0.0, 1.0);
        if order.is_buy {
            let y_sub = order.size * alpha;
            let y_norm = order.size * (1.0 - alpha);
            let out_sub = if y_sub > MIN_TRADE_SIZE {
                amm_sub.quote_buy_x(y_sub)
            } else {
                0.0
            };
            let out_norm = if y_norm > MIN_TRADE_SIZE {
                amm_norm.quote_buy_x(y_norm)
            } else {
                0.0
            };
            out_sub + out_norm
        } else {
            let total_x = order.size / fair_price.max(1e-12);
            let x_sub = total_x * alpha;
            let x_norm = total_x * (1.0 - alpha);
            let out_sub = if x_sub > MIN_TRADE_SIZE {
                amm_sub.quote_sell_x(x_sub)
            } else {
                0.0
            };
            let out_norm = if x_norm > MIN_TRADE_SIZE {
                amm_norm.quote_sell_x(x_norm)
            } else {
                0.0
            };
            out_sub + out_norm
        }
    }

    fn brute_force_best_output(
        order: &RetailOrder,
        fair_price: f64,
        sub_swap: SwapFn,
        norm_swap: SwapFn,
        sub_reserves: (f64, f64),
        norm_reserves: (f64, f64),
    ) -> f64 {
        let mut amm_sub = BpfAmm::new_native(
            sub_swap,
            None,
            sub_reserves.0,
            sub_reserves.1,
            "sub".to_string(),
        );
        let mut amm_norm = BpfAmm::new_native(
            norm_swap,
            None,
            norm_reserves.0,
            norm_reserves.1,
            "norm".to_string(),
        );

        let mut best = 0.0_f64;
        for i in 0..=BRUTE_FORCE_STEPS {
            let alpha = i as f64 / BRUTE_FORCE_STEPS as f64;
            let out = quote_total_output(order, fair_price, alpha, &mut amm_sub, &mut amm_norm);
            if out > best {
                best = out;
            }
        }
        best
    }

    fn run_router_once(
        order: &RetailOrder,
        fair_price: f64,
        sub_swap: SwapFn,
        norm_swap: SwapFn,
        sub_reserves: (f64, f64),
        norm_reserves: (f64, f64),
    ) -> f64 {
        let router = OrderRouter::new();
        let mut amm_sub = BpfAmm::new_native(
            sub_swap,
            None,
            sub_reserves.0,
            sub_reserves.1,
            "sub".to_string(),
        );
        let mut amm_norm = BpfAmm::new_native(
            norm_swap,
            None,
            norm_reserves.0,
            norm_reserves.1,
            "norm".to_string(),
        );
        let trades = router.route_order(order, &mut amm_sub, &mut amm_norm, fair_price);
        total_output_from_trades(order, &trades)
    }

    fn assert_close_to_optimal(
        router_output: f64,
        brute_force_optimum: f64,
        relative_tolerance: f64,
        context: &str,
    ) {
        let tolerance = brute_force_optimum * relative_tolerance + 1e-8;
        assert!(
            router_output + tolerance >= brute_force_optimum,
            "{context}: router output too far from optimum (router={}, brute={}, tolerance={})",
            router_output,
            brute_force_optimum,
            tolerance
        );
    }

    #[test]
    fn router_buy_search_is_close_to_bruteforce_across_diverse_curves() {
        let mut rng = Pcg64::seed_from_u64(7);
        let curve_set: [SwapFn; 5] = [
            normalizer_swap,
            zero_fee_swap,
            low_fee_swap,
            starter_fee_swap,
            high_fee_swap,
        ];

        for case_idx in 0..220 {
            let sub_swap = *curve_set.choose(&mut rng).unwrap();
            let norm_swap = *curve_set.choose(&mut rng).unwrap();
            let sub_rx = rng.gen_range(20.0..400.0);
            let sub_price = rng.gen_range(35.0..220.0);
            let norm_rx = sub_rx * rng.gen_range(0.6..1.6);
            let norm_price = sub_price * rng.gen_range(0.6..1.6);
            let sub_ry = sub_rx * sub_price;
            let norm_ry = norm_rx * norm_price;
            let fair_price = ((sub_price + norm_price) * 0.5) * rng.gen_range(0.7..1.3);
            let order = RetailOrder {
                is_buy: true,
                size: rng.gen_range(0.5..2_500.0),
            };

            let router_output = run_router_once(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );
            let brute = brute_force_best_output(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );

            assert_close_to_optimal(
                router_output,
                brute,
                DIVERSE_CURVE_TOLERANCE,
                &format!("buy case {case_idx}"),
            );
        }
    }

    #[test]
    fn router_sell_search_is_close_to_bruteforce_across_diverse_curves() {
        let mut rng = Pcg64::seed_from_u64(11);
        let curve_set: [SwapFn; 5] = [
            normalizer_swap,
            zero_fee_swap,
            low_fee_swap,
            starter_fee_swap,
            high_fee_swap,
        ];

        for case_idx in 0..220 {
            let sub_swap = *curve_set.choose(&mut rng).unwrap();
            let norm_swap = *curve_set.choose(&mut rng).unwrap();
            let sub_rx = rng.gen_range(20.0..400.0);
            let sub_price = rng.gen_range(35.0..220.0);
            let norm_rx = sub_rx * rng.gen_range(0.6..1.6);
            let norm_price = sub_price * rng.gen_range(0.6..1.6);
            let sub_ry = sub_rx * sub_price;
            let norm_ry = norm_rx * norm_price;
            let fair_price = ((sub_price + norm_price) * 0.5) * rng.gen_range(0.7..1.3);
            let order = RetailOrder {
                is_buy: false,
                size: rng.gen_range(0.5..2_500.0),
            };

            let router_output = run_router_once(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );
            let brute = brute_force_best_output(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );

            assert_close_to_optimal(
                router_output,
                brute,
                DIVERSE_CURVE_TOLERANCE,
                &format!("sell case {case_idx}"),
            );
        }
    }

    #[test]
    fn router_finds_near_optimal_split_on_endpoint_dominance_regimes() {
        let mut rng = Pcg64::seed_from_u64(99);

        for case_idx in 0..240 {
            let sub_rx = rng.gen_range(15.0..280.0);
            let sub_price = rng.gen_range(20.0..250.0);
            let norm_rx = sub_rx * rng.gen_range(0.7..1.4);
            let norm_price = sub_price * rng.gen_range(0.7..1.4);
            let sub_ry = sub_rx * sub_price;
            let norm_ry = norm_rx * norm_price;
            let fair_price = ((sub_price + norm_price) * 0.5) * rng.gen_range(0.8..1.2);
            let order = RetailOrder {
                is_buy: rng.gen_bool(0.5),
                size: rng.gen_range(1.0..3_000.0),
            };
            let (sub_swap, norm_swap): (SwapFn, SwapFn) = if rng.gen_bool(0.5) {
                (high_fee_swap, zero_fee_swap)
            } else {
                (zero_fee_swap, high_fee_swap)
            };

            let router_output = run_router_once(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );
            let brute = brute_force_best_output(
                &order,
                fair_price,
                sub_swap,
                norm_swap,
                (sub_rx, sub_ry),
                (norm_rx, norm_ry),
            );

            assert_close_to_optimal(
                router_output,
                brute,
                ENDPOINT_REGIME_TOLERANCE,
                &format!("endpoint regime case {case_idx}"),
            );
        }
    }
}
