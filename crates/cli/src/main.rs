mod commands;
mod output;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "prop-amm", about = "Prop AMM Challenge CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build program (native for simulation, BPF for submission)
    Build {
        /// Path to the .rs source file
        file: String,
    },
    /// Validate a program (convexity, monotonicity, CU)
    Validate {
        /// Path to the .rs source file
        file: String,
    },
    /// Run simulation batch
    Run {
        /// Path to the .rs source file
        file: String,
        /// Number of simulations
        #[arg(long, default_value = "1000")]
        simulations: u32,
        /// Number of steps per simulation
        #[arg(long, default_value = "10000")]
        steps: u32,
        /// Number of parallel workers (0 = auto)
        #[arg(long, default_value = "0")]
        workers: usize,
        /// Use BPF runtime instead of native (slower, for validation)
        #[arg(long)]
        bpf: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { file } => commands::build::run(&file),
        Commands::Validate { file } => commands::validate::run(&file),
        Commands::Run {
            file,
            simulations,
            steps,
            workers,
            bpf,
        } => commands::run::run(&file, simulations, steps, workers, bpf),
    }
}
