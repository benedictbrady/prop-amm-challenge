use rayon::prelude::*;

use prop_amm_executor::{AfterSwapFn, BpfProgram, SwapFn};
use prop_amm_shared::config::{HyperparameterVariance, SimulationConfig};
use prop_amm_shared::result::{BatchResult, SimResult};

use crate::engine;

pub fn run_batch(
    submission_program: BpfProgram,
    normalizer_program: BpfProgram,
    configs: Vec<SimulationConfig>,
    n_workers: Option<usize>,
) -> anyhow::Result<BatchResult> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers.unwrap_or_else(|| rayon::current_num_threads().min(8)))
        .build()?;

    let results: Result<Vec<SimResult>, _> = pool.install(|| {
        configs
            .par_iter()
            .map(|config| {
                let sub = submission_program.clone();
                let norm = normalizer_program.clone();
                engine::run_simulation(sub, norm, config)
            })
            .collect()
    });

    Ok(BatchResult::from_results(results?))
}

pub fn run_batch_native(
    submission_fn: SwapFn,
    submission_after_swap: Option<AfterSwapFn>,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    configs: Vec<SimulationConfig>,
    n_workers: Option<usize>,
) -> anyhow::Result<BatchResult> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers.unwrap_or_else(|| rayon::current_num_threads().min(8)))
        .build()?;

    let results: Result<Vec<SimResult>, _> = pool.install(|| {
        configs
            .par_iter()
            .map(|config| {
                engine::run_simulation_native(
                    submission_fn,
                    submission_after_swap,
                    normalizer_fn,
                    normalizer_after_swap,
                    config,
                )
            })
            .collect()
    });

    Ok(BatchResult::from_results(results?))
}

pub fn run_default_batch(
    submission_program: BpfProgram,
    normalizer_program: BpfProgram,
    n_sims: u32,
    n_steps: u32,
    n_workers: Option<usize>,
) -> anyhow::Result<BatchResult> {
    let variance = HyperparameterVariance::default();
    let mut base = SimulationConfig::default();
    base.n_steps = n_steps;
    let configs: Vec<_> = (0..n_sims)
        .map(|i| variance.apply(&base, i as u64))
        .collect();
    run_batch(submission_program, normalizer_program, configs, n_workers)
}

pub fn run_default_batch_mixed(
    submission_program: BpfProgram,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    n_sims: u32,
    n_steps: u32,
    n_workers: Option<usize>,
) -> anyhow::Result<BatchResult> {
    let variance = HyperparameterVariance::default();
    let mut base = SimulationConfig::default();
    base.n_steps = n_steps;
    let configs: Vec<_> = (0..n_sims)
        .map(|i| variance.apply(&base, i as u64))
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers.unwrap_or_else(|| rayon::current_num_threads().min(8)))
        .build()?;

    let results: Result<Vec<SimResult>, _> = pool.install(|| {
        configs
            .par_iter()
            .map(|config| {
                let sub = submission_program.clone();
                engine::run_simulation_mixed(sub, normalizer_fn, normalizer_after_swap, config)
            })
            .collect()
    });

    Ok(BatchResult::from_results(results?))
}

pub fn run_default_batch_native(
    submission_fn: SwapFn,
    submission_after_swap: Option<AfterSwapFn>,
    normalizer_fn: SwapFn,
    normalizer_after_swap: Option<AfterSwapFn>,
    n_sims: u32,
    n_steps: u32,
    n_workers: Option<usize>,
) -> anyhow::Result<BatchResult> {
    let variance = HyperparameterVariance::default();
    let mut base = SimulationConfig::default();
    base.n_steps = n_steps;
    let configs: Vec<_> = (0..n_sims)
        .map(|i| variance.apply(&base, i as u64))
        .collect();
    run_batch_native(
        submission_fn,
        submission_after_swap,
        normalizer_fn,
        normalizer_after_swap,
        configs,
        n_workers,
    )
}
