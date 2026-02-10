use crate::amm::BpfAmm;
use crate::retail::RetailOrder;
use crate::search::{DeterministicSearch, SearchContext, SearchPhase};

pub struct RoutedTrade {
    pub is_submission: bool,
    pub amm_buys_x: bool,
    pub amount_x: f64,
    pub amount_y: f64,
}

const MIN_SPLIT_SIZE: f64 = 0.001;

pub struct OrderRouter;

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
        search: &DeterministicSearch,
        step: u32,
        event: u32,
    ) -> Vec<RoutedTrade> {
        if order.is_buy {
            self.route_buy(
                order.size,
                amm_sub,
                amm_norm,
                search,
                SearchContext {
                    step,
                    event,
                    phase: SearchPhase::RouterBuy,
                },
            )
        } else {
            let total_x = order.size / fair_price;
            self.route_sell(
                total_x,
                amm_sub,
                amm_norm,
                search,
                SearchContext {
                    step,
                    event,
                    phase: SearchPhase::RouterSell,
                },
            )
        }
    }

    fn route_buy(
        &self,
        total_y: f64,
        amm_sub: &mut BpfAmm,
        amm_norm: &mut BpfAmm,
        search: &DeterministicSearch,
        context: SearchContext,
    ) -> Vec<RoutedTrade> {
        let split = search.optimize(0.0, total_y, total_y * 0.5, context, |y_sub| {
            let y_norm = (total_y - y_sub).max(0.0);
            let x_sub = if y_sub > MIN_SPLIT_SIZE {
                amm_sub.quote_buy_x(y_sub)
            } else {
                0.0
            };
            let x_norm = if y_norm > MIN_SPLIT_SIZE {
                amm_norm.quote_buy_x(y_norm)
            } else {
                0.0
            };
            x_sub + x_norm
        });

        let mut trades = Vec::new();
        let y_sub = split.input.clamp(0.0, total_y);
        let y_norm = (total_y - y_sub).max(0.0);

        if y_sub > MIN_SPLIT_SIZE {
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
        if y_norm > MIN_SPLIT_SIZE {
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
        search: &DeterministicSearch,
        context: SearchContext,
    ) -> Vec<RoutedTrade> {
        let split = search.optimize(0.0, total_x, total_x * 0.5, context, |x_sub| {
            let x_norm = (total_x - x_sub).max(0.0);
            let y_sub = if x_sub > MIN_SPLIT_SIZE {
                amm_sub.quote_sell_x(x_sub)
            } else {
                0.0
            };
            let y_norm = if x_norm > MIN_SPLIT_SIZE {
                amm_norm.quote_sell_x(x_norm)
            } else {
                0.0
            };
            y_sub + y_norm
        });

        let mut trades = Vec::new();
        let x_sub = split.input.clamp(0.0, total_x);
        let x_norm = (total_x - x_sub).max(0.0);

        if x_sub > MIN_SPLIT_SIZE {
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
        if x_norm > MIN_SPLIT_SIZE {
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
}
