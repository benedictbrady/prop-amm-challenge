use prop_amm_executor::{AfterSwapFn, BpfProgram, SwapFn};
use prop_amm_shared::config::SimulationConfig;
use prop_amm_shared::result::SimResult;

use crate::amm::BpfAmm;
use crate::arbitrageur::Arbitrageur;
use crate::price_process::GBMPriceProcess;
use crate::retail::RetailTrader;
use crate::router::OrderRouter;

fn run_sim_inner(
    mut amm_sub: BpfAmm,
    mut amm_norm: BpfAmm,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let mut price = GBMPriceProcess::new(
        config.initial_price,
        config.gbm_mu,
        config.gbm_sigma,
        config.gbm_dt,
        config.seed,
    );
    let mut retail = RetailTrader::new(
        config.retail_arrival_rate,
        config.retail_mean_size,
        config.retail_size_sigma,
        config.retail_buy_prob,
        config.seed.wrapping_add(1),
    );
    let arb = Arbitrageur::new(config.min_arb_profit);
    let router = OrderRouter::new();

    let mut submission_edge = 0.0_f64;

    for _ in 0..config.n_steps {
        let fair_price = price.step();

        if let Some(result) = arb.execute_arb(&mut amm_sub, fair_price) {
            submission_edge += result.edge;
        }
        arb.execute_arb(&mut amm_norm, fair_price);

        let orders = retail.generate_orders();
        for order in &orders {
            let trades = router.route_order(order, &mut amm_sub, &mut amm_norm, fair_price);
            for trade in trades {
                if trade.is_submission {
                    let trade_edge = if trade.amm_buys_x {
                        trade.amount_x * fair_price - trade.amount_y
                    } else {
                        trade.amount_y - trade.amount_x * fair_price
                    };
                    submission_edge += trade_edge;
                }
            }
        }
    }

    Ok(SimResult {
        seed: config.seed,
        submission_edge,
    })
}

/// Run simulation with BPF programs (slow, for validation)
pub fn run_simulation(
    submission_program: BpfProgram,
    normalizer_program: BpfProgram,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new(submission_program, config.initial_x, config.initial_y, "submission".to_string());
    let amm_norm = BpfAmm::new(normalizer_program, config.initial_x, config.initial_y, "normalizer".to_string());
    run_sim_inner(amm_sub, amm_norm, config)
}

/// Run simulation with native swap functions (fast, for production)
pub fn run_simulation_native(
    submission_fn: SwapFn,
    submission_after_swap: Option<AfterSwapFn>,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new_native(submission_fn, submission_after_swap, config.initial_x, config.initial_y, "submission".to_string());
    let amm_norm = BpfAmm::new_native(normalizer_fn, normalizer_after_swap, config.initial_x, config.initial_y, "normalizer".to_string());
    run_sim_inner(amm_sub, amm_norm, config)
}

/// Run simulation with BPF submission + native normalizer (mixed mode)
pub fn run_simulation_mixed(
    submission_program: BpfProgram,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    config: &SimulationConfig,
) -> anyhow::Result<SimResult> {
    let amm_sub = BpfAmm::new(submission_program, config.initial_x, config.initial_y, "submission".to_string());
    let amm_norm = BpfAmm::new_native(normalizer_fn, normalizer_after_swap, config.initial_x, config.initial_y, "normalizer".to_string());
    run_sim_inner(amm_sub, amm_norm, config)
}
