use std::f64::consts::PI;

const PRIMARY_PROBES: usize = 8;
const REFINEMENT_FACTORS: [f64; 5] = [0.7, 0.85, 1.0, 1.15, 1.3];

#[repr(u64)]
#[derive(Clone, Copy)]
pub enum SearchPhase {
    ArbBuy = 1,
    ArbSell = 2,
    RouterBuy = 3,
    RouterSell = 4,
}

#[derive(Clone, Copy)]
pub struct SearchContext {
    pub step: u32,
    pub event: u32,
    pub phase: SearchPhase,
}

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub input: f64,
    pub score: f64,
}

pub struct DeterministicSearch {
    seed: u64,
    retail_mean_size: f64,
    retail_size_sigma: f64,
    retail_mu_ln: f64,
}

impl DeterministicSearch {
    pub fn new(seed: u64, retail_mean_size: f64, retail_size_sigma: f64) -> Self {
        let sigma = retail_size_sigma.max(0.01);
        let mean = retail_mean_size.max(0.01);
        let mu_ln = mean.ln() - 0.5 * sigma * sigma;
        Self {
            seed,
            retail_mean_size: mean,
            retail_size_sigma: sigma,
            retail_mu_ln: mu_ln,
        }
    }

    #[inline]
    pub fn retail_mean_size(&self) -> f64 {
        self.retail_mean_size
    }

    pub fn optimize<F>(
        &self,
        min_input: f64,
        max_input: f64,
        center_hint: f64,
        context: SearchContext,
        mut objective: F,
    ) -> SearchResult
    where
        F: FnMut(f64) -> f64,
    {
        if max_input <= min_input {
            return SearchResult {
                input: min_input.max(0.0),
                score: objective(min_input.max(0.0)),
            };
        }

        let mut best_input = min_input;
        let mut best_score = f64::NEG_INFINITY;

        for probe_idx in 0..PRIMARY_PROBES {
            let input = self.probe_input(probe_idx, min_input, max_input, center_hint, context);
            let score = objective(input);
            if score > best_score {
                best_score = score;
                best_input = input;
            }
        }

        // Always include boundaries so the optimizer can still pick one-pool routing.
        for boundary in [min_input, max_input] {
            let score = objective(boundary);
            if score > best_score {
                best_score = score;
                best_input = boundary;
            }
        }

        for (i, factor) in REFINEMENT_FACTORS.iter().enumerate() {
            let probe = (best_input * factor).clamp(min_input, max_input);
            let score = objective(probe);
            if score > best_score {
                best_score = score;
                best_input = probe;
            }

            // Small deterministic dither around refinement probes avoids deterministic fingerprints.
            let jitter = 0.985 + 0.03 * self.unit(context, 200 + i as u32, 0);
            let probe_jittered = (probe * jitter).clamp(min_input, max_input);
            let score_jittered = objective(probe_jittered);
            if score_jittered > best_score {
                best_score = score_jittered;
                best_input = probe_jittered;
            }
        }

        SearchResult {
            input: best_input,
            score: best_score,
        }
    }

    fn probe_input(
        &self,
        probe_idx: usize,
        min_input: f64,
        max_input: f64,
        center_hint: f64,
        context: SearchContext,
    ) -> f64 {
        // The first probe is always sampled from the retail size distribution.
        if probe_idx == 0 {
            return self.retail_like_size(
                center_hint,
                min_input,
                max_input,
                context,
                probe_idx as u32,
            );
        }

        let mix = self.unit(context, probe_idx as u32, 0);
        if mix < 0.8 {
            self.retail_like_size(center_hint, min_input, max_input, context, probe_idx as u32)
        } else {
            // Deterministic high-tail exploration to keep arb search effective.
            let u = self.unit(context, probe_idx as u32, 1);
            (min_input + (max_input - min_input) * u.powf(0.35)).clamp(min_input, max_input)
        }
    }

    fn retail_like_size(
        &self,
        center_hint: f64,
        min_input: f64,
        max_input: f64,
        context: SearchContext,
        probe_idx: u32,
    ) -> f64 {
        let u1 = self.unit(context, probe_idx, 2).clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.unit(context, probe_idx, 3);

        // Box-Muller transform for deterministic normal draws.
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let base_retail = (self.retail_mu_ln + self.retail_size_sigma * z).exp();

        let center = center_hint.max(0.01);
        let scale = (center / self.retail_mean_size).clamp(0.25, 8.0);
        (base_retail * scale).clamp(min_input, max_input)
    }

    #[inline]
    fn unit(&self, context: SearchContext, probe_idx: u32, stream: u32) -> f64 {
        let mixed = splitmix64(
            self.seed
                ^ ((context.step as u64).wrapping_mul(0x9E3779B97F4A7C15))
                ^ ((context.event as u64).wrapping_mul(0xBF58476D1CE4E5B9))
                ^ ((context.phase as u64).wrapping_mul(0x94D049BB133111EB))
                ^ ((probe_idx as u64).wrapping_mul(0xD6E8FEB86659FD93))
                ^ ((stream as u64).wrapping_mul(0xA5A3564E27F1E123)),
        );
        let bits = mixed >> 11;
        bits as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
