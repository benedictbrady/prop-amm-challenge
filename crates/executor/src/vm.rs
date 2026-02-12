use solana_rbpf::{
    aligned_memory::AlignedMemory,
    ebpf,
    memory_region::{MemoryMapping, MemoryRegion},
    vm::EbpfVm,
};

use crate::loader::{BpfProgram, ExecutorError};
use crate::syscalls::SyscallContext;
use prop_amm_shared::instruction::{AFTER_SWAP_SIZE, STORAGE_SIZE, SWAP_INSTRUCTION_SIZE};

/// Solana input buffer layout for 0 accounts:
/// [0..8]   u64 num_accounts = 0
/// [8..16]  u64 instruction_data_len
/// [16..]   instruction_data (up to AFTER_SWAP_SIZE bytes)
/// [..]     program_id (32 bytes, zeros)
const INPUT_BUF_SIZE: usize = 8 + 8 + AFTER_SWAP_SIZE + 32; // 1106

pub struct BpfExecutor {
    program: BpfProgram,
    input_buf: Vec<u8>,
    stack: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    heap: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    context: SyscallContext,
}

impl BpfExecutor {
    pub fn new(program: BpfProgram) -> Self {
        let config = program.executable().get_config();
        let input_buf = vec![0u8; INPUT_BUF_SIZE];

        Self {
            stack: AlignedMemory::zero_filled(config.stack_size()),
            heap: AlignedMemory::zero_filled(32 * 1024),
            program,
            input_buf,
            context: SyscallContext::new(100_000),
        }
    }

    fn run_vm(&mut self, instr_data_len: usize) -> Result<(), ExecutorError> {
        // Write instruction data length
        self.input_buf[8..16].copy_from_slice(&(instr_data_len as u64).to_le_bytes());

        // Reset context flags without reallocating storage Vec.
        self.context.reset(100_000);

        let executable = self.program.executable();
        let loader = self.program.loader();
        let config = executable.get_config();
        let sbpf_version = executable.get_sbpf_version();
        let stack_len = self.stack.len();

        let regions: Vec<MemoryRegion> = vec![
            executable.get_ro_region(),
            MemoryRegion::new_writable(self.stack.as_slice_mut(), ebpf::MM_STACK_START),
            MemoryRegion::new_writable(self.heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new_writable(&mut self.input_buf, ebpf::MM_INPUT_START),
        ];

        let memory_mapping = MemoryMapping::new(regions, config, sbpf_version)
            .map_err(|e| ExecutorError::Execution(e.to_string()))?;

        let mut vm = EbpfVm::new(
            loader.clone(),
            sbpf_version,
            &mut self.context,
            memory_mapping,
            stack_len,
        );

        let use_interpreter = !self.program.jit_available();
        let (_instruction_count, result) = vm.execute_program(executable, use_interpreter);

        let result: Result<u64, _> = result.into();
        result.map_err(|e| ExecutorError::Execution(e.to_string()))?;

        Ok(())
    }

    pub fn execute(
        &mut self,
        side: u8,
        amount: u64,
        rx: u64,
        ry: u64,
        storage: &[u8],
    ) -> Result<u64, ExecutorError> {
        self.input_buf.fill(0);

        // Write instruction data: [side(1)][amount(8)][rx(8)][ry(8)][storage(1024)]
        self.input_buf[16] = side;
        self.input_buf[17..25].copy_from_slice(&amount.to_le_bytes());
        self.input_buf[25..33].copy_from_slice(&rx.to_le_bytes());
        self.input_buf[33..41].copy_from_slice(&ry.to_le_bytes());
        let copy_len = storage.len().min(STORAGE_SIZE);
        self.input_buf[41..41 + copy_len].copy_from_slice(&storage[..copy_len]);
        if copy_len < STORAGE_SIZE {
            self.input_buf[41 + copy_len..41 + STORAGE_SIZE].fill(0);
        }

        self.run_vm(SWAP_INSTRUCTION_SIZE)?;

        if !self.context.has_return_data {
            return Err(ExecutorError::NoReturnData);
        }

        Ok(u64::from_le_bytes(self.context.return_data))
    }

    pub fn execute_after_swap(
        &mut self,
        side: u8,
        input_amount: u64,
        output_amount: u64,
        rx: u64,
        ry: u64,
        step: u64,
        storage: &mut [u8],
    ) -> Result<(), ExecutorError> {
        self.input_buf.fill(0);

        // Write after_swap instruction data:
        // [tag=2(1)][side(1)][input(8)][output(8)][rx(8)][ry(8)][step(8)][storage(1024)]
        self.input_buf[16] = 2; // tag
        self.input_buf[17] = side;
        self.input_buf[18..26].copy_from_slice(&input_amount.to_le_bytes());
        self.input_buf[26..34].copy_from_slice(&output_amount.to_le_bytes());
        self.input_buf[34..42].copy_from_slice(&rx.to_le_bytes());
        self.input_buf[42..50].copy_from_slice(&ry.to_le_bytes());
        self.input_buf[50..58].copy_from_slice(&step.to_le_bytes());
        let copy_len = storage.len().min(STORAGE_SIZE);
        self.input_buf[58..58 + copy_len].copy_from_slice(&storage[..copy_len]);
        if copy_len < STORAGE_SIZE {
            self.input_buf[58 + copy_len..58 + STORAGE_SIZE].fill(0);
        }

        self.run_vm(AFTER_SWAP_SIZE)?;

        if self.context.has_storage_update {
            let out_len = storage.len().min(STORAGE_SIZE);
            storage[..out_len].copy_from_slice(&self.context.storage_data[..out_len]);
        }

        Ok(())
    }
}
