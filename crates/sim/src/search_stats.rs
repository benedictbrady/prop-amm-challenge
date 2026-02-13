use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PROP_AMM_SEARCH_STATS").is_some())
}

#[derive(Debug, Clone, Copy)]
pub struct SearchStatsSnapshot {
    pub arb_bracket_calls: u64,
    pub arb_bracket_evals: u64,
    pub arb_golden_calls: u64,
    pub arb_golden_iters: u64,
    pub arb_golden_evals: u64,
    pub arb_early_stop_amount_tol: u64,
    pub router_calls: u64,
    pub router_golden_iters: u64,
    pub router_evals: u64,
    pub router_early_stop_rel_gap: u64,
}

static ARB_BRACKET_CALLS: AtomicU64 = AtomicU64::new(0);
static ARB_BRACKET_EVALS: AtomicU64 = AtomicU64::new(0);
static ARB_GOLDEN_CALLS: AtomicU64 = AtomicU64::new(0);
static ARB_GOLDEN_ITERS: AtomicU64 = AtomicU64::new(0);
static ARB_GOLDEN_EVALS: AtomicU64 = AtomicU64::new(0);
static ARB_EARLY_STOP_AMOUNT_TOL: AtomicU64 = AtomicU64::new(0);

static ROUTER_CALLS: AtomicU64 = AtomicU64::new(0);
static ROUTER_GOLDEN_ITERS: AtomicU64 = AtomicU64::new(0);
static ROUTER_EVALS: AtomicU64 = AtomicU64::new(0);
static ROUTER_EARLY_STOP_REL_GAP: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    ARB_BRACKET_CALLS.store(0, Ordering::Relaxed);
    ARB_BRACKET_EVALS.store(0, Ordering::Relaxed);
    ARB_GOLDEN_CALLS.store(0, Ordering::Relaxed);
    ARB_GOLDEN_ITERS.store(0, Ordering::Relaxed);
    ARB_GOLDEN_EVALS.store(0, Ordering::Relaxed);
    ARB_EARLY_STOP_AMOUNT_TOL.store(0, Ordering::Relaxed);
    ROUTER_CALLS.store(0, Ordering::Relaxed);
    ROUTER_GOLDEN_ITERS.store(0, Ordering::Relaxed);
    ROUTER_EVALS.store(0, Ordering::Relaxed);
    ROUTER_EARLY_STOP_REL_GAP.store(0, Ordering::Relaxed);
}

pub fn snapshot_if_enabled() -> Option<SearchStatsSnapshot> {
    if !enabled() {
        return None;
    }
    Some(SearchStatsSnapshot {
        arb_bracket_calls: ARB_BRACKET_CALLS.load(Ordering::Relaxed),
        arb_bracket_evals: ARB_BRACKET_EVALS.load(Ordering::Relaxed),
        arb_golden_calls: ARB_GOLDEN_CALLS.load(Ordering::Relaxed),
        arb_golden_iters: ARB_GOLDEN_ITERS.load(Ordering::Relaxed),
        arb_golden_evals: ARB_GOLDEN_EVALS.load(Ordering::Relaxed),
        arb_early_stop_amount_tol: ARB_EARLY_STOP_AMOUNT_TOL.load(Ordering::Relaxed),
        router_calls: ROUTER_CALLS.load(Ordering::Relaxed),
        router_golden_iters: ROUTER_GOLDEN_ITERS.load(Ordering::Relaxed),
        router_evals: ROUTER_EVALS.load(Ordering::Relaxed),
        router_early_stop_rel_gap: ROUTER_EARLY_STOP_REL_GAP.load(Ordering::Relaxed),
    })
}

#[inline]
pub(crate) fn inc_arb_bracket_call() {
    if enabled() {
        ARB_BRACKET_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_arb_bracket_eval() {
    if enabled() {
        ARB_BRACKET_EVALS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_arb_golden_call() {
    if enabled() {
        ARB_GOLDEN_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_arb_golden_iter() {
    if enabled() {
        ARB_GOLDEN_ITERS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_arb_golden_eval() {
    if enabled() {
        ARB_GOLDEN_EVALS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_arb_early_stop_amount_tol() {
    if enabled() {
        ARB_EARLY_STOP_AMOUNT_TOL.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_router_call() {
    if enabled() {
        ROUTER_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_router_iter() {
    if enabled() {
        ROUTER_GOLDEN_ITERS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_router_eval() {
    if enabled() {
        ROUTER_EVALS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn inc_router_early_stop_rel_gap() {
    if enabled() {
        ROUTER_EARLY_STOP_REL_GAP.fetch_add(1, Ordering::Relaxed);
    }
}
