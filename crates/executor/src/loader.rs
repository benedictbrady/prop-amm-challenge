use std::sync::Arc;

use solana_rbpf::{
    elf::Executable,
    program::{BuiltinFunction, BuiltinProgram, FunctionRegistry},
    verifier::RequisiteVerifier,
    vm::Config,
};

use crate::syscalls::{
    SyscallAbort, SyscallContext, SyscallLog, SyscallMemcmp, SyscallMemcpy, SyscallMemmove,
    SyscallMemset, SyscallSetReturnData, SyscallSetStorage,
};

#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("ELF loading failed: {0}")]
    ElfLoad(String),
    #[error("Verification failed: {0}")]
    Verification(String),
    #[error("JIT compilation failed: {0}")]
    JitCompilation(String),
    #[error("Execution failed: {0}")]
    Execution(String),
    #[error("No return data")]
    NoReturnData,
    #[error("Program aborted")]
    Aborted,
}

#[derive(Clone)]
pub struct BpfProgram {
    executable: Arc<Executable<SyscallContext>>,
    loader: Arc<BuiltinProgram<SyscallContext>>,
    jit_available: bool,
}

impl BpfProgram {
    pub fn load(elf_bytes: &[u8]) -> Result<Self, ExecutorError> {
        let mut function_registry = FunctionRegistry::<BuiltinFunction<SyscallContext>>::default();

        function_registry
            .register_function_hashed(*b"sol_set_return_data", SyscallSetReturnData::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_log_", SyscallLog::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"abort", SyscallAbort::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_set_storage", SyscallSetStorage::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_memcpy_", SyscallMemcpy::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_memmove_", SyscallMemmove::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_memcmp_", SyscallMemcmp::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;
        function_registry
            .register_function_hashed(*b"sol_memset_", SyscallMemset::vm)
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;

        let loader = Arc::new(BuiltinProgram::new_loader(
            Config::default(),
            function_registry,
        ));

        #[allow(unused_mut)]
        let mut executable = Executable::<SyscallContext>::from_elf(elf_bytes, loader.clone())
            .map_err(|e| ExecutorError::ElfLoad(e.to_string()))?;

        executable
            .verify::<RequisiteVerifier>()
            .map_err(|e| ExecutorError::Verification(e.to_string()))?;

        // JIT compile on x86_64 for near-native speed from BPF programs.
        // On ARM (macOS dev), falls back to interpreter — use native mode instead.
        #[allow(unused_mut)]
        let mut jit_available = false;
        #[cfg(all(not(target_os = "windows"), target_arch = "x86_64"))]
        {
            if executable.jit_compile().is_ok() {
                jit_available = true;
            }
        }

        Ok(Self {
            executable: Arc::new(executable),
            loader,
            jit_available,
        })
    }

    pub fn executable(&self) -> &Arc<Executable<SyscallContext>> {
        &self.executable
    }

    pub fn loader(&self) -> &Arc<BuiltinProgram<SyscallContext>> {
        &self.loader
    }

    pub fn jit_available(&self) -> bool {
        self.jit_available
    }
}
