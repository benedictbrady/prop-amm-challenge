use std::path::{Path, PathBuf};
use std::process::Command;

const BUILD_DIR: &str = ".build";

const CARGO_TOML: &str = r#"[package]
name = "user_program"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]

[dependencies]
pinocchio = "0.7"

[features]
no-entrypoint = []
"#;

pub fn ensure_build_dir() -> anyhow::Result<PathBuf> {
    let build_dir = PathBuf::from(BUILD_DIR);
    std::fs::create_dir_all(build_dir.join("src"))?;

    let cargo_path = build_dir.join("Cargo.toml");
    if !cargo_path.exists() {
        std::fs::write(&cargo_path, CARGO_TOML)?;
    }

    Ok(build_dir)
}

pub fn compile_native(rs_file: &str) -> anyhow::Result<PathBuf> {
    let rs_path = Path::new(rs_file);
    if !rs_path.exists() {
        anyhow::bail!("File not found: {}", rs_file);
    }

    let build_dir = ensure_build_dir()?;
    std::fs::copy(rs_path, build_dir.join("src/lib.rs"))?;

    let status = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg(build_dir.join("Cargo.toml"))
        .arg("--features")
        .arg("no-entrypoint")
        .status()?;

    if !status.success() {
        anyhow::bail!("Native build failed");
    }

    find_native_lib(&build_dir)
}

pub fn compile_bpf(rs_file: &str) -> anyhow::Result<PathBuf> {
    let rs_path = Path::new(rs_file);
    if !rs_path.exists() {
        anyhow::bail!("File not found: {}", rs_file);
    }

    let build_dir = ensure_build_dir()?;
    std::fs::copy(rs_path, build_dir.join("src/lib.rs"))?;

    let status = Command::new("cargo")
        .arg("build-sbf")
        .arg("--manifest-path")
        .arg(build_dir.join("Cargo.toml"))
        .status()?;

    if !status.success() {
        anyhow::bail!("BPF build failed");
    }

    find_bpf_so(&build_dir)
}

fn find_native_lib(build_dir: &Path) -> anyhow::Result<PathBuf> {
    let release_dir = build_dir.join("target").join("release");
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    if let Ok(entries) = std::fs::read_dir(&release_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("lib") && name.ends_with(ext) {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!(
        "No native library found in {}/target/release/",
        build_dir.display()
    )
}

fn find_bpf_so(build_dir: &Path) -> anyhow::Result<PathBuf> {
    let deploy_dir = build_dir.join("target").join("deploy");

    if let Ok(entries) = std::fs::read_dir(&deploy_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".so") {
                return Ok(entry.path());
            }
        }
    }

    anyhow::bail!(
        "No BPF .so found in {}/target/deploy/",
        build_dir.display()
    )
}
