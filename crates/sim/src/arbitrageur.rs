use crate::amm::BpfAmm;
use crate::search::{DeterministicSearch, SearchContext, SearchPhase};

const PRICE_TOLERANCE_PCT: f64 = 0.0001;
const MIN_TRADE_SIZE: f64 = 0.001;
const MAX_ARB_CLIPS: u32 = 4;
const MAX_RETAIL_SCALE: f64 = 8.0;

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

    pub fn execute_arb(
        &self,
        amm: &mut BpfAmm,
        fair_price: f64,
        search: &DeterministicSearch,
        step: u32,
        event: u32,
    ) -> Option<ArbResult> {
        let spot = amm.spot_price();

        if spot < fair_price * (1.0 - PRICE_TOLERANCE_PCT) {
            self.arb_buy_x(amm, fair_price, search, step, event)
        } else if spot > fair_price * (1.0 + PRICE_TOLERANCE_PCT) {
            self.arb_sell_x(amm, fair_price, search, step, event)
        } else {
            None
        }
    }

    fn arb_buy_x(
        &self,
        amm: &mut BpfAmm,
        fair_price: f64,
        search: &DeterministicSearch,
        step: u32,
        event: u32,
    ) -> Option<ArbResult> {
        let max_y = (search.retail_mean_size() * MAX_RETAIL_SCALE).min(amm.reserve_y * 0.5);
        if max_y <= MIN_TRADE_SIZE {
            return None;
        }

        let mut total_input_y = 0.0_f64;
        let mut total_output_x = 0.0_f64;

        for clip in 0..MAX_ARB_CLIPS {
            let context = SearchContext {
                step,
                event: event.wrapping_add(clip),
                phase: SearchPhase::ArbBuy,
            };

            let candidate = search.optimize(
                MIN_TRADE_SIZE,
                max_y,
                search.retail_mean_size(),
                context,
                |input_y| {
                    let output_x = amm.quote_buy_x(input_y);
                    output_x * fair_price - input_y
                },
            );

            if candidate.score <= self.min_arb_profit {
                break;
            }

            let output_x = amm.execute_buy_x(candidate.input);
            if output_x <= 0.0 {
                break;
            }

            total_input_y += candidate.input;
            total_output_x += output_x;

            if amm.spot_price() >= fair_price * (1.0 - PRICE_TOLERANCE_PCT * 0.25) {
                break;
            }
        }

        if total_output_x <= 0.0 {
            return None;
        }

        Some(ArbResult {
            amm_buys_x: false,
            amount_x: total_output_x,
            amount_y: total_input_y,
            edge: total_input_y - total_output_x * fair_price,
        })
    }

    fn arb_sell_x(
        &self,
        amm: &mut BpfAmm,
        fair_price: f64,
        search: &DeterministicSearch,
        step: u32,
        event: u32,
    ) -> Option<ArbResult> {
        let max_x =
            (search.retail_mean_size() / fair_price * MAX_RETAIL_SCALE).min(amm.reserve_x * 0.5);
        if max_x <= MIN_TRADE_SIZE {
            return None;
        }

        let mut total_input_x = 0.0_f64;
        let mut total_output_y = 0.0_f64;

        for clip in 0..MAX_ARB_CLIPS {
            let context = SearchContext {
                step,
                event: event.wrapping_add(clip),
                phase: SearchPhase::ArbSell,
            };

            let candidate = search.optimize(
                MIN_TRADE_SIZE,
                max_x,
                search.retail_mean_size() / fair_price,
                context,
                |input_x| {
                    let output_y = amm.quote_sell_x(input_x);
                    output_y - input_x * fair_price
                },
            );

            if candidate.score <= self.min_arb_profit {
                break;
            }

            let output_y = amm.execute_sell_x(candidate.input);
            if output_y <= 0.0 {
                break;
            }

            total_input_x += candidate.input;
            total_output_y += output_y;

            if amm.spot_price() <= fair_price * (1.0 + PRICE_TOLERANCE_PCT * 0.25) {
                break;
            }
        }

        if total_output_y <= 0.0 {
            return None;
        }

        Some(ArbResult {
            amm_buys_x: true,
            amount_x: total_input_x,
            amount_y: total_output_y,
            edge: total_input_x * fair_price - total_output_y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Arbitrageur;
    use crate::amm::BpfAmm;
    use crate::search::DeterministicSearch;
    use prop_amm_shared::normalizer::compute_swap as normalizer_swap;

    fn test_amm() -> BpfAmm {
        BpfAmm::new_native(normalizer_swap, None, 100.0, 10_000.0, "test".to_string())
    }

    fn test_search() -> DeterministicSearch {
        DeterministicSearch::new(42, 20.0, 1.2)
    }

    #[test]
    fn min_arb_profit_blocks_profitable_trade_when_threshold_is_higher() {
        let fair_price = 101.0;
        let search = test_search();

        let mut amm_without_floor = test_amm();
        let no_floor = Arbitrageur::new(0.0);
        let result = no_floor
            .execute_arb(&mut amm_without_floor, fair_price, &search, 0, 0)
            .expect("expected profitable arbitrage");
        let realized_profit = -result.edge;
        assert!(
            realized_profit > 0.0,
            "arb should produce positive arb profit"
        );

        let mut amm_with_floor = test_amm();
        let floor = Arbitrageur::new(realized_profit + 1e-9);
        assert!(
            floor
                .execute_arb(&mut amm_with_floor, fair_price, &search, 0, 0)
                .is_none(),
            "trade should be skipped when profit ({realized_profit}) is below threshold"
        );
    }
}
