use crate::amm::BpfAmm;
use crate::retail::RetailOrder;

pub struct RoutedTrade {
    pub is_submission: bool,
    pub amm_buys_x: bool,
    pub amount_x: f64,
    pub amount_y: f64,
}

const GRID_POINTS: usize = 11;
const MIN_TRADE_SIZE: f64 = 0.001;
const CURVE_EPS: f64 = 1e-9;

pub struct OrderRouter;

#[derive(Clone, Copy)]
struct QuotePoint {
    alpha: f64,
    in_sub: f64,
    in_norm: f64,
    out_sub: f64,
    out_norm: f64,
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
        let mut points = Vec::with_capacity(GRID_POINTS);

        for i in 0..GRID_POINTS {
            let alpha = i as f64 / (GRID_POINTS - 1) as f64;
            let y_sub = total_y * alpha;
            let y_norm = total_y * (1.0 - alpha);

            let x_sub = if y_sub > MIN_TRADE_SIZE {
                amm_sub.quote_buy_x(y_sub)
            } else {
                0.0
            };
            let x_norm = if y_norm > MIN_TRADE_SIZE {
                amm_norm.quote_buy_x(y_norm)
            } else {
                0.0
            };

            points.push(QuotePoint {
                alpha,
                in_sub: y_sub,
                in_norm: y_norm,
                out_sub: x_sub,
                out_norm: x_norm,
            });
        }

        let sub_curve: Vec<(f64, f64)> = points.iter().map(|p| (p.in_sub, p.out_sub)).collect();
        let norm_curve: Vec<(f64, f64)> = points.iter().map(|p| (p.in_norm, p.out_norm)).collect();
        let sub_valid = Self::is_valid_curve(&sub_curve, amm_sub.reserve_x);
        let norm_valid = Self::is_valid_curve(&norm_curve, amm_norm.reserve_x);

        let best = points
            .iter()
            .filter(|p| Self::split_is_allowed(p, sub_valid, norm_valid))
            .max_by(|a, b| {
                let a_total = a.out_sub + a.out_norm;
                let b_total = b.out_sub + b.out_norm;
                a_total.total_cmp(&b_total)
            })
            .copied();

        let Some(best) = best else {
            return Vec::new();
        };

        if (best.alpha - 0.0).abs() < CURVE_EPS && !norm_valid {
            return Vec::new();
        }
        if (best.alpha - 1.0).abs() < CURVE_EPS && !sub_valid {
            return Vec::new();
        }

        let mut trades = Vec::new();
        let y_sub = best.in_sub;
        let y_norm = best.in_norm;

        if y_sub > MIN_TRADE_SIZE && sub_valid {
            let x_out = amm_sub.execute_buy_x_quoted(y_sub, best.out_sub);
            if x_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: true,
                    amm_buys_x: false,
                    amount_x: x_out,
                    amount_y: y_sub,
                });
            }
        }
        if y_norm > MIN_TRADE_SIZE && norm_valid {
            let x_out = amm_norm.execute_buy_x_quoted(y_norm, best.out_norm);
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
        let mut points = Vec::with_capacity(GRID_POINTS);

        for i in 0..GRID_POINTS {
            let alpha = i as f64 / (GRID_POINTS - 1) as f64;
            let x_sub = total_x * alpha;
            let x_norm = total_x * (1.0 - alpha);

            let y_sub = if x_sub > MIN_TRADE_SIZE {
                amm_sub.quote_sell_x(x_sub)
            } else {
                0.0
            };
            let y_norm = if x_norm > MIN_TRADE_SIZE {
                amm_norm.quote_sell_x(x_norm)
            } else {
                0.0
            };

            points.push(QuotePoint {
                alpha,
                in_sub: x_sub,
                in_norm: x_norm,
                out_sub: y_sub,
                out_norm: y_norm,
            });
        }

        let sub_curve: Vec<(f64, f64)> = points.iter().map(|p| (p.in_sub, p.out_sub)).collect();
        let norm_curve: Vec<(f64, f64)> = points.iter().map(|p| (p.in_norm, p.out_norm)).collect();
        let sub_valid = Self::is_valid_curve(&sub_curve, amm_sub.reserve_y);
        let norm_valid = Self::is_valid_curve(&norm_curve, amm_norm.reserve_y);

        let best = points
            .iter()
            .filter(|p| Self::split_is_allowed(p, sub_valid, norm_valid))
            .max_by(|a, b| {
                let a_total = a.out_sub + a.out_norm;
                let b_total = b.out_sub + b.out_norm;
                a_total.total_cmp(&b_total)
            })
            .copied();

        let Some(best) = best else {
            return Vec::new();
        };

        if (best.alpha - 0.0).abs() < CURVE_EPS && !norm_valid {
            return Vec::new();
        }
        if (best.alpha - 1.0).abs() < CURVE_EPS && !sub_valid {
            return Vec::new();
        }

        let mut trades = Vec::new();
        let x_sub = best.in_sub;
        let x_norm = best.in_norm;

        if x_sub > MIN_TRADE_SIZE && sub_valid {
            let y_out = amm_sub.execute_sell_x_quoted(x_sub, best.out_sub);
            if y_out > 0.0 {
                trades.push(RoutedTrade {
                    is_submission: true,
                    amm_buys_x: true,
                    amount_x: x_sub,
                    amount_y: y_out,
                });
            }
        }
        if x_norm > MIN_TRADE_SIZE && norm_valid {
            let y_out = amm_norm.execute_sell_x_quoted(x_norm, best.out_norm);
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

    fn split_is_allowed(point: &QuotePoint, sub_valid: bool, norm_valid: bool) -> bool {
        if !sub_valid && point.in_sub > MIN_TRADE_SIZE {
            return false;
        }
        if !norm_valid && point.in_norm > MIN_TRADE_SIZE {
            return false;
        }
        true
    }

    fn is_valid_curve(points: &[(f64, f64)], max_output: f64) -> bool {
        if !max_output.is_finite() || max_output <= 0.0 {
            return false;
        }

        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut prev_input: Option<f64> = None;
        let mut prev_output: Option<f64> = None;
        let mut prev_slope = f64::INFINITY;

        for (input, output) in &sorted_points {
            if *input < 0.0 || !input.is_finite() || !output.is_finite() {
                return false;
            }
            if *output < -CURVE_EPS || *output > max_output + CURVE_EPS {
                return false;
            }

            if let Some(last_output) = prev_output {
                if *output + CURVE_EPS < last_output {
                    return false;
                }
            }

            if let (Some(last_input), Some(last_output)) = (prev_input, prev_output) {
                let delta_input = *input - last_input;
                if delta_input > CURVE_EPS {
                    let slope = (*output - last_output) / delta_input;
                    if slope < -CURVE_EPS {
                        return false;
                    }
                    // Quotes are quantized to 1e-9 token units; permit tiny slope increases
                    // consistent with one-tick rounding noise.
                    let slope_rounding_tol = (2.0e-9_f64 / delta_input).max(CURVE_EPS);
                    if slope > prev_slope + slope_rounding_tol {
                        return false;
                    }
                    prev_slope = slope;
                }
            }

            prev_input = Some(*input);
            prev_output = Some(*output);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::{OrderRouter, GRID_POINTS, MIN_TRADE_SIZE};
    use crate::amm::BpfAmm;
    use crate::retail::RetailOrder;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CALL_COUNTS: OnceLock<Mutex<HashMap<(u8, u64, u64, u64), u32>>> = OnceLock::new();

    fn quote_bait_swap(data: &[u8]) -> u64 {
        if data.len() < 25 {
            return 0;
        }
        let side = data[0];
        let input = u64::from_le_bytes(data[1..9].try_into().unwrap());
        let rx = u64::from_le_bytes(data[9..17].try_into().unwrap());
        let ry = u64::from_le_bytes(data[17..25].try_into().unwrap());
        let key = (side, input, rx, ry);

        let counts = CALL_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut counts = counts.lock().unwrap();
        let entry = counts.entry(key).or_insert(0);
        *entry += 1;

        if *entry == 1 {
            normalizer_swap(data).saturating_add(100)
        } else {
            1
        }
    }

    fn high_fee_swap(data: &[u8]) -> u64 {
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

        let k = rx * ry;
        match side {
            0 => {
                let net = input * 50 / 1000; // 95% fee
                let new_ry = ry + net;
                rx.saturating_sub((k + new_ry - 1) / new_ry) as u64
            }
            1 => {
                let net = input * 50 / 1000; // 95% fee
                let new_rx = rx + net;
                ry.saturating_sub((k + new_rx - 1) / new_rx) as u64
            }
            _ => 0,
        }
    }

    fn normalizer_buy_curve(reserve_x: f64, reserve_y: f64, total_y: f64) -> Vec<(f64, f64)> {
        let mut amm_norm = BpfAmm::new_native(
            normalizer_swap,
            None,
            reserve_x,
            reserve_y,
            "norm".to_string(),
        );
        let mut points = Vec::with_capacity(GRID_POINTS);
        for i in 0..GRID_POINTS {
            let alpha = i as f64 / (GRID_POINTS - 1) as f64;
            let in_norm = total_y * (1.0 - alpha);
            let out_norm = if in_norm > MIN_TRADE_SIZE {
                amm_norm.quote_buy_x(in_norm)
            } else {
                0.0
            };
            points.push((in_norm, out_norm));
        }
        points
    }

    fn normalizer_sell_curve(reserve_x: f64, reserve_y: f64, total_x: f64) -> Vec<(f64, f64)> {
        let mut amm_norm = BpfAmm::new_native(
            normalizer_swap,
            None,
            reserve_x,
            reserve_y,
            "norm".to_string(),
        );
        let mut points = Vec::with_capacity(GRID_POINTS);
        for i in 0..GRID_POINTS {
            let alpha = i as f64 / (GRID_POINTS - 1) as f64;
            let in_norm = total_x * (1.0 - alpha);
            let out_norm = if in_norm > MIN_TRADE_SIZE {
                amm_norm.quote_sell_x(in_norm)
            } else {
                0.0
            };
            points.push((in_norm, out_norm));
        }
        points
    }

    #[test]
    fn router_executes_committed_quote_output() {
        let counts = CALL_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
        counts.lock().unwrap().clear();

        let router = OrderRouter::new();
        let mut amm_sub =
            BpfAmm::new_native(quote_bait_swap, None, 100.0, 10_000.0, "sub".to_string());
        let mut amm_norm =
            BpfAmm::new_native(normalizer_swap, None, 100.0, 10_000.0, "norm".to_string());
        let order = RetailOrder {
            is_buy: true,
            size: 20.0,
        };

        let trades = router.route_order(&order, &mut amm_sub, &mut amm_norm, 100.0);
        let sub_trade = trades
            .iter()
            .find(|t| t.is_submission)
            .expect("submission leg should execute");

        // With quote-commit, execution must use the selected quote output and not re-quote.
        assert!(
            sub_trade.amount_x > 0.05,
            "unexpectedly tiny execution output: {}",
            sub_trade.amount_x
        );
    }

    #[test]
    fn curve_validator_accepts_descending_input_points() {
        let descending = vec![
            (20.0, 1.0),
            (15.0, 0.85),
            (10.0, 0.65),
            (5.0, 0.38),
            (0.0, 0.0),
        ];
        assert!(
            OrderRouter::is_valid_curve(&descending, 100.0),
            "descending input order should be normalized before validation"
        );
    }

    #[test]
    fn router_prefers_normalizer_over_extreme_fee_submission() {
        let router = OrderRouter::new();
        let mut amm_sub =
            BpfAmm::new_native(high_fee_swap, None, 100.0, 10_000.0, "sub".to_string());
        let mut amm_norm =
            BpfAmm::new_native(normalizer_swap, None, 100.0, 10_000.0, "norm".to_string());

        let order = RetailOrder {
            is_buy: true,
            size: 200.0,
        };

        let trades = router.route_order(&order, &mut amm_sub, &mut amm_norm, 100.0);
        assert!(
            trades.iter().any(|t| !t.is_submission),
            "normalizer leg should be selected when submission quotes are much worse"
        );
        assert!(
            !trades.iter().any(|t| t.is_submission),
            "submission leg should not be selected for this case"
        );
    }

    #[test]
    fn normalizer_curves_are_valid_across_many_buy_regimes() {
        let reserve_x_values = [25.0, 50.0, 100.0, 250.0, 500.0];
        let reserve_y_values = [2_500.0, 5_000.0, 10_000.0, 25_000.0, 50_000.0];
        let order_sizes_y = [1.0, 5.0, 20.0, 100.0, 300.0, 1_000.0];

        for reserve_x in reserve_x_values {
            for reserve_y in reserve_y_values {
                for total_y in order_sizes_y {
                    let curve = normalizer_buy_curve(reserve_x, reserve_y, total_y);
                    assert!(
                        OrderRouter::is_valid_curve(&curve, reserve_x),
                        "normalizer buy curve invalid (rx={}, ry={}, total_y={})",
                        reserve_x,
                        reserve_y,
                        total_y
                    );
                }
            }
        }
    }

    #[test]
    fn normalizer_curves_are_valid_across_many_sell_regimes() {
        let reserve_x_values = [25.0, 50.0, 100.0, 250.0, 500.0];
        let reserve_y_values = [2_500.0, 5_000.0, 10_000.0, 25_000.0, 50_000.0];
        let order_sizes_x = [0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

        for reserve_x in reserve_x_values {
            for reserve_y in reserve_y_values {
                for total_x in order_sizes_x {
                    let curve = normalizer_sell_curve(reserve_x, reserve_y, total_x);
                    assert!(
                        OrderRouter::is_valid_curve(&curve, reserve_y),
                        "normalizer sell curve invalid (rx={}, ry={}, total_x={})",
                        reserve_x,
                        reserve_y,
                        total_x
                    );
                }
            }
        }
    }

    #[test]
    fn normalizer_router_path_never_returns_empty_trades() {
        let router = OrderRouter::new();
        let fair_prices = [50.0, 80.0, 100.0, 140.0, 200.0];
        let order_sizes = [5.0, 20.0, 100.0, 300.0, 1_000.0];

        for fair_price in fair_prices {
            for size in order_sizes {
                for is_buy in [true, false] {
                    let mut amm_sub = BpfAmm::new_native(
                        normalizer_swap,
                        None,
                        100.0,
                        10_000.0,
                        "sub".to_string(),
                    );
                    let mut amm_norm = BpfAmm::new_native(
                        normalizer_swap,
                        None,
                        100.0,
                        10_000.0,
                        "norm".to_string(),
                    );
                    let order = RetailOrder { is_buy, size };

                    let trades =
                        router.route_order(&order, &mut amm_sub, &mut amm_norm, fair_price);
                    assert!(
                        !trades.is_empty(),
                        "expected non-empty routed trades (is_buy={}, size={}, fair_price={})",
                        is_buy,
                        size,
                        fair_price
                    );
                    for trade in &trades {
                        assert!(trade.amount_x.is_finite() && trade.amount_x > 0.0);
                        assert!(trade.amount_y.is_finite() && trade.amount_y > 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn high_fee_submission_gets_no_flow_across_many_orders() {
        let router = OrderRouter::new();
        let fair_prices = [80.0, 100.0, 130.0];
        let order_sizes = [2.0, 10.0, 50.0, 200.0, 500.0];

        for fair_price in fair_prices {
            for size in order_sizes {
                for is_buy in [true, false] {
                    let mut amm_sub =
                        BpfAmm::new_native(high_fee_swap, None, 100.0, 10_000.0, "sub".to_string());
                    let mut amm_norm = BpfAmm::new_native(
                        normalizer_swap,
                        None,
                        100.0,
                        10_000.0,
                        "norm".to_string(),
                    );
                    let order = RetailOrder { is_buy, size };

                    let trades =
                        router.route_order(&order, &mut amm_sub, &mut amm_norm, fair_price);
                    assert!(
                        trades.iter().all(|t| !t.is_submission),
                        "submission unexpectedly received flow (is_buy={}, size={}, fair_price={})",
                        is_buy,
                        size,
                        fair_price
                    );
                }
            }
        }
    }

    #[test]
    fn normalizer_curve_validation_stays_stable_under_randomized_regimes() {
        let mut rng = Pcg64::seed_from_u64(1337);
        for _ in 0..200 {
            let reserve_x = rng.gen_range(20.0..600.0);
            let mid_price = rng.gen_range(40.0..250.0);
            let reserve_y = reserve_x * mid_price;
            let order_y = rng.gen_range(1.0..2_000.0);
            let order_x = rng.gen_range(0.01..25.0);

            let buy_curve = normalizer_buy_curve(reserve_x, reserve_y, order_y);
            assert!(
                OrderRouter::is_valid_curve(&buy_curve, reserve_x),
                "randomized buy curve invalid (rx={}, ry={}, order_y={})",
                reserve_x,
                reserve_y,
                order_y
            );

            let sell_curve = normalizer_sell_curve(reserve_x, reserve_y, order_x);
            assert!(
                OrderRouter::is_valid_curve(&sell_curve, reserve_y),
                "randomized sell curve invalid (rx={}, ry={}, order_x={})",
                reserve_x,
                reserve_y,
                order_x
            );
        }
    }
}
