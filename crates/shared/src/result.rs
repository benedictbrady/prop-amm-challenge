#[derive(Debug, Clone)]
pub struct SimResult {
    pub seed: u64,
    pub submission_edge: f64,
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub results: Vec<SimResult>,
    pub total_edge: f64,
}

impl BatchResult {
    pub fn from_results(results: Vec<SimResult>) -> Self {
        let total_edge = results.iter().map(|r| r.submission_edge).sum();
        Self {
            results,
            total_edge,
        }
    }

    pub fn n_sims(&self) -> usize {
        self.results.len()
    }

    pub fn avg_edge(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            self.total_edge / self.results.len() as f64
        }
    }
}
