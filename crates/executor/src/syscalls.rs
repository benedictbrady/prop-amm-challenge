use solana_rbpf::{
    declare_builtin_function,
    error::EbpfError,
    memory_region::{AccessType, MemoryMapping},
    vm::ContextObject,
};

use prop_amm_shared::instruction::STORAGE_SIZE;

pub struct SyscallContext {
    pub return_data: [u8; 8],
    pub has_return_data: bool,
    pub storage_data: Vec<u8>,
    pub has_storage_update: bool,
    remaining: u64,
}

impl SyscallContext {
    pub fn new(remaining: u64) -> Self {
        Self {
            return_data: [0u8; 8],
            has_return_data: false,
            storage_data: vec![0u8; STORAGE_SIZE],
            has_storage_update: false,
            remaining,
        }
    }

    /// Reset for reuse without reallocating the storage Vec.
    pub fn reset(&mut self, remaining: u64) {
        self.has_return_data = false;
        self.has_storage_update = false;
        self.remaining = remaining;
    }
}

impl ContextObject for SyscallContext {
    fn trace(&mut self, _state: [u64; 12]) {}

    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }

    fn get_remaining(&self) -> u64 {
        self.remaining
    }
}

declare_builtin_function!(
    /// BPF program calls this to set return data.
    /// arg1 = vm address of data, arg2 = length, arg3 = unused program_id addr
    SyscallSetReturnData,
    fn rust(
        context_object: &mut SyscallContext,
        addr: u64,
        len: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len > 8 {
            return Err(Box::new(EbpfError::AccessViolation(
                AccessType::Load,
                addr,
                len,
                "input",
            )));
        }
        let host_addr: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, addr, len).into();
        let host_addr = host_addr?;
        let slice = unsafe { std::slice::from_raw_parts(host_addr as *const u8, len as usize) };
        context_object.return_data = [0u8; 8];
        context_object.return_data[..len as usize].copy_from_slice(slice);
        context_object.has_return_data = true;
        Ok(0)
    }
);

declare_builtin_function!(
    /// No-op log syscall
    SyscallLog,
    fn rust(
        _context_object: &mut SyscallContext,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(0)
    }
);

declare_builtin_function!(
    /// Abort syscall - returns an error
    SyscallAbort,
    fn rust(
        _context_object: &mut SyscallContext,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Err("program aborted".into())
    }
);

declare_builtin_function!(
    /// BPF program calls this to write updated storage after afterSwap.
    /// arg1 = vm address of data, arg2 = length (must be <= STORAGE_SIZE)
    SyscallSetStorage,
    fn rust(
        context_object: &mut SyscallContext,
        addr: u64,
        len: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len > STORAGE_SIZE as u64 {
            return Err(Box::new(EbpfError::AccessViolation(
                AccessType::Load,
                addr,
                len,
                "input",
            )));
        }
        let host_addr: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, addr, len).into();
        let host_addr = host_addr?;
        let slice = unsafe { std::slice::from_raw_parts(host_addr as *const u8, len as usize) };
        context_object.storage_data[..len as usize].copy_from_slice(slice);
        // Zero remaining bytes if partial write
        if (len as usize) < STORAGE_SIZE {
            context_object.storage_data[len as usize..].fill(0);
        }
        context_object.has_storage_update = true;
        Ok(0)
    }
);
