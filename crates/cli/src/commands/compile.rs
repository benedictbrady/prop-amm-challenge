use std::path::{Path, PathBuf};
use std::process::Command;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

const BUILD_RUNS_DIR: &str = ".build/runs";
const FORBID_UNSAFE: &str = "#![forbid(unsafe_code)]\n";

const CARGO_TOML: &str = r#"[package]
name = "user_program"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]

[dependencies]
pinocchio = "0.7"
wincode = { version = "0.4", default-features = false, features = ["derive"] }
prop-amm-submission-sdk = { path = "../crates/submission-sdk" }

[features]
no-entrypoint = []
"#;

fn cargo_toml_with_sdk_path() -> String {
    CARGO_TOML.replace(
        "path = \"../crates/submission-sdk\"",
        "path = \"../../../crates/submission-sdk\"",
    )
}

pub fn ensure_build_dir(safe_source: &str) -> anyhow::Result<PathBuf> {
    let mut hasher = DefaultHasher::new();
    safe_source.hash(&mut hasher);
    CARGO_TOML.hash(&mut hasher);
    let build_key = format!("{:016x}", hasher.finish());

    let build_dir = PathBuf::from(BUILD_RUNS_DIR).join(build_key);
    std::fs::create_dir_all(build_dir.join("src"))?;

    let cargo_toml = cargo_toml_with_sdk_path();
    let cargo_path = build_dir.join("Cargo.toml");
    let should_write = match std::fs::read_to_string(&cargo_path) {
        Ok(existing) => existing != cargo_toml,
        Err(_) => true,
    };
    if should_write {
        std::fs::write(&cargo_path, cargo_toml)?;
    }

    let source_path = build_dir.join("src/lib.rs");
    let source_bytes = safe_source.as_bytes();
    let should_write_source = match std::fs::read(&source_path) {
        Ok(existing) => existing != source_bytes,
        Err(_) => true,
    };
    if should_write_source {
        std::fs::write(source_path, source_bytes)?;
    }

    Ok(build_dir)
}

pub fn compile_native(rs_file: &str) -> anyhow::Result<PathBuf> {
    let rs_path = Path::new(rs_file);
    if !rs_path.exists() {
        anyhow::bail!("File not found: {}", rs_file);
    }

    let safe_source = make_safe_submission_source(rs_path)?;
    let build_dir = ensure_build_dir(&safe_source)?;

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

    let safe_source = make_safe_submission_source(rs_path)?;
    let build_dir = ensure_build_dir(&safe_source)?;

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

fn make_safe_submission_source(rs_path: &Path) -> anyhow::Result<String> {
    let source = std::fs::read_to_string(rs_path)?;
    let safe_source = if source.starts_with(FORBID_UNSAFE) {
        source
    } else {
        let mut wrapped = String::with_capacity(FORBID_UNSAFE.len() + source.len());
        wrapped.push_str(FORBID_UNSAFE);
        wrapped.push_str(&source);
        wrapped
    };

    Ok(safe_source)
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

    anyhow::bail!("No BPF .so found in {}/target/deploy/", build_dir.display())
}
