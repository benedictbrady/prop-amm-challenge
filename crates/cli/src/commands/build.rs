use std::process::Command;

pub fn run(path: &str) -> anyhow::Result<()> {
    let manifest = format!("{}/Cargo.toml", path);

    // Build native cdylib for simulation
    println!("Building native library...");
    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--features")
        .arg("no-entrypoint")
        .status()?;

    if !status.success() {
        anyhow::bail!("Native build failed");
    }

    // Find the native library
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let output = Command::new("find")
        .arg(format!("{}/target/release", path))
        .arg("-maxdepth")
        .arg("1")
        .arg("-name")
        .arg(format!("*.{}", ext))
        .output()?;
    let native_path = String::from_utf8_lossy(&output.stdout);
    let native_path = native_path.lines().next().unwrap_or("").trim();
    if !native_path.is_empty() {
        println!("  Native: {}", native_path);
    }

    // Build BPF for submission
    println!("Building BPF program...");
    let status = Command::new("cargo")
        .arg("build-sbf")
        .arg("--manifest-path")
        .arg(&manifest)
        .status()?;

    if !status.success() {
        anyhow::bail!("BPF build failed");
    }

    let output = Command::new("find")
        .arg(path)
        .arg("-name")
        .arg("*.so")
        .arg("-path")
        .arg("*/deploy/*")
        .output()?;
    let bpf_path = String::from_utf8_lossy(&output.stdout);
    let bpf_path = bpf_path.lines().next().unwrap_or("").trim();
    if !bpf_path.is_empty() {
        println!("  BPF:    {}", bpf_path);
    }

    println!("\nRun locally:");
    if !native_path.is_empty() {
        println!("  prop-amm run {}", native_path);
    }
    if !bpf_path.is_empty() {
        println!("\nSubmit to API:");
        println!("  Upload {}", bpf_path);
    }

    Ok(())
}
