use crate::amm::BpfAmm;

pub struct ArbResult {
    pub amm_buys_x: bool,
    pub amount_x: f64,
    pub amount_y: f64,
    pub edge: f64,
}

pub struct Arbitrageur {
    min_arb_profit: f64,
}

impl Arbitrageur {
    pub fn new(min_arb_profit: f64) -> Self {
        Self {
            min_arb_profit: min_arb_profit.max(0.0),
        }
    }

    pub fn execute_arb(&self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let spot = amm.spot_price();

        if spot < fair_price * 0.9999 {
            self.arb_buy_x(amm, fair_price)
        } else if spot > fair_price * 1.0001 {
            self.arb_sell_x(amm, fair_price)
        } else {
            None
        }
    }

    fn arb_buy_x(&self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let mut lo = 0.0_f64;
        let mut hi = amm.reserve_y * 0.5;

        for _ in 0..12 {
            let mid = (lo + hi) / 2.0;
            let eps = mid * 0.001 + 0.001;
            let out_lo = amm.quote_buy_x(mid);
            let out_hi = amm.quote_buy_x(mid + eps);
            let marginal_x = (out_hi - out_lo) / eps;
            if marginal_x * fair_price > 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let optimal_y = (lo + hi) / 2.0;
        if optimal_y < 0.001 {
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

    fn arb_sell_x(&self, amm: &mut BpfAmm, fair_price: f64) -> Option<ArbResult> {
        let mut lo = 0.0_f64;
        let mut hi = amm.reserve_x * 0.5;

        for _ in 0..12 {
            let mid = (lo + hi) / 2.0;
            let eps = mid * 0.001 + 0.001;
            let out_lo = amm.quote_sell_x(mid);
            let out_hi = amm.quote_sell_x(mid + eps);
            let marginal_y = (out_hi - out_lo) / eps;
            if marginal_y > fair_price {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let optimal_x = (lo + hi) / 2.0;
        if optimal_x < 0.001 {
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
        let no_floor = Arbitrageur::new(0.0);
        let result = no_floor
            .execute_arb(&mut amm_without_floor, fair_price)
            .expect("expected profitable arbitrage");
        let realized_profit = -result.edge;
        assert!(
            realized_profit > 0.0,
            "arb should produce positive arb profit"
        );

        let mut amm_with_floor = test_amm();
        let floor = Arbitrageur::new(realized_profit + 1e-9);
        assert!(
            floor.execute_arb(&mut amm_with_floor, fair_price).is_none(),
            "trade should be skipped when profit ({realized_profit}) is below threshold"
        );
    }
}
