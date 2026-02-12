use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

// Baseline simulation parameters
pub const BASELINE_STEPS: u32 = 10_000;
pub const BASELINE_SIMS: u32 = 1_000;
pub const INITIAL_PRICE: f64 = 100.0;
pub const INITIAL_X: f64 = 100.0;
pub const INITIAL_Y: f64 = 10_000.0;
pub const GBM_MU: f64 = 0.0;
pub const GBM_SIGMA: f64 = 0.000945; // baseline point estimate
pub const GBM_DT: f64 = 1.0;
pub const RETAIL_ARRIVAL_RATE: f64 = 0.8; // midpoint of [0.4, 1.2]
pub const RETAIL_MEAN_SIZE: f64 = 20.0; // midpoint of [12, 28]
pub const RETAIL_SIZE_SIGMA: f64 = 1.2;
pub const RETAIL_BUY_PROB: f64 = 0.5;
pub const MIN_ARB_PROFIT: f64 = 0.01; // 1 cent in quote token (Y)

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub n_steps: u32,
    pub initial_price: f64,
    pub initial_x: f64,
    pub initial_y: f64,
    pub gbm_mu: f64,
    pub gbm_sigma: f64,
    pub gbm_dt: f64,
    pub retail_arrival_rate: f64,
    pub retail_mean_size: f64,
    pub retail_size_sigma: f64,
    pub retail_buy_prob: f64,
    pub min_arb_profit: f64,
    pub seed: u64,
    pub norm_fee_bps: u16,
    pub norm_liquidity_mult: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_steps: BASELINE_STEPS,
            initial_price: INITIAL_PRICE,
            initial_x: INITIAL_X,
            initial_y: INITIAL_Y,
            gbm_mu: GBM_MU,
            gbm_sigma: GBM_SIGMA,
            gbm_dt: GBM_DT,
            retail_arrival_rate: RETAIL_ARRIVAL_RATE,
            retail_mean_size: RETAIL_MEAN_SIZE,
            retail_size_sigma: RETAIL_SIZE_SIGMA,
            retail_buy_prob: RETAIL_BUY_PROB,
            min_arb_profit: MIN_ARB_PROFIT,
            seed: 0,
            norm_fee_bps: 30,
            norm_liquidity_mult: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HyperparameterVariance {
    pub gbm_sigma_min: f64,
    pub gbm_sigma_max: f64,
    pub retail_arrival_rate_min: f64,
    pub retail_arrival_rate_max: f64,
    pub retail_mean_size_min: f64,
    pub retail_mean_size_max: f64,
    pub norm_fee_bps_min: u16,
    pub norm_fee_bps_max: u16,
    pub norm_liquidity_mult_min: f64,
    pub norm_liquidity_mult_max: f64,
}

impl Default for HyperparameterVariance {
    fn default() -> Self {
        Self {
            gbm_sigma_min: 0.0001,
            gbm_sigma_max: 0.007,
            retail_arrival_rate_min: 0.4,
            retail_arrival_rate_max: 1.2,
            retail_mean_size_min: 12.0,
            retail_mean_size_max: 28.0,
            norm_fee_bps_min: 30,
            norm_fee_bps_max: 80,
            norm_liquidity_mult_min: 0.4,
            norm_liquidity_mult_max: 2.0,
        }
    }
}

impl HyperparameterVariance {
    pub fn apply(&self, base: &SimulationConfig, seed: u64) -> SimulationConfig {
        let mut rng = Pcg64::seed_from_u64(seed);
        // Original 3 draws first — order preserved for seed reproducibility
        let gbm_sigma = rng.gen_range(self.gbm_sigma_min..self.gbm_sigma_max);
        let retail_arrival_rate = rng.gen_range(self.retail_arrival_rate_min..self.retail_arrival_rate_max);
        let retail_mean_size = rng.gen_range(self.retail_mean_size_min..self.retail_mean_size_max);
        // New draws appended
        let norm_fee_bps = rng.gen_range(self.norm_fee_bps_min..=self.norm_fee_bps_max);
        let norm_liquidity_mult = rng.gen_range(self.norm_liquidity_mult_min..self.norm_liquidity_mult_max);
        SimulationConfig {
            gbm_sigma,
            retail_arrival_rate,
            retail_mean_size,
            norm_fee_bps,
            norm_liquidity_mult,
            seed,
            ..base.clone()
        }
    }

    pub fn generate_configs(&self, n: u32) -> Vec<SimulationConfig> {
        let base = SimulationConfig::default();
        (0..n).map(|i| self.apply(&base, i as u64)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationConfig;

    #[test]
    fn default_min_arb_profit_is_one_cent() {
        let config = SimulationConfig::default();
        assert!((config.min_arb_profit - 0.01).abs() < 1e-12);
    }
}
