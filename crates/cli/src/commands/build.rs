use super::compile;

pub fn run(file: &str) -> anyhow::Result<()> {
    println!("Building native library...");
    let native_path = compile::compile_native(file)?;
    println!("  Native: {}", native_path.display());

    println!("Building BPF program...");
    let bpf_path = compile::compile_bpf(file)?;
    println!("  BPF:    {}", bpf_path.display());

    println!("\nRun locally:");
    println!("  prop-amm run {}", file);
    println!("\nRun via BPF:");
    println!("  prop-amm run {} --bpf", file);
    println!("\nSubmit to API:");
    println!("  Upload {}", bpf_path.display());

    Ok(())
}
