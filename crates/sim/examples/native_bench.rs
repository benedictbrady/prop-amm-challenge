use prop_amm_shared::normalizer::compute_swap as normalizer_swap;
use prop_amm_sim::runner;
use std::time::Instant;

fn main() {
    // Starter (50bp) as native function
    fn starter_swap(data: &[u8]) -> u64 {
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
                let net = input * 950 / 1000;
                let new_ry = ry + net;
                rx.saturating_sub((k + new_ry - 1) / new_ry) as u64
            }
            1 => {
                let net = input * 950 / 1000;
                let new_rx = rx + net;
                ry.saturating_sub((k + new_rx - 1) / new_rx) as u64
            }
            _ => 0,
        }
    }

    println!("Running 1000 simulations / 10k steps (native)...");
    let start = Instant::now();
    let result = runner::run_default_batch_native(
        starter_swap,
        None,
        normalizer_swap,
        None,
        1000,
        10_000,
        None,
    )
    .unwrap();
    let elapsed = start.elapsed();

    println!("========================================");
    println!("  Simulations: {}", result.n_sims());
    println!("  Time:        {:.2}s", elapsed.as_secs_f64());
    println!("  Avg edge:    {:.2}", result.avg_edge());
    println!("  Total edge:  {:.2}", result.total_edge);
    println!("========================================");
}
