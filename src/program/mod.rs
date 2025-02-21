//! Individual program representation and execution.
//!
//! This module defines how individual programs in the genetic programming system
//! are represented and executed. Each program consists of:
//!
//! - A sequence of register-based instructions
//! - Variable registers for calculations
//! - Constant registers with fixed values
//!
//! Programs take a single input and produce a single output, making them
//! suitable for evolving mathematical functions.
//!
//! # Example
//!
//! ```
//! use lgp::program::Program;
//!
//! // Create a program with 5 instructions
//! let mut program = Program::new(5);
//!
//! // Run the program with input 2.0
//! let output = program.run(2.0);
//! ```

mod operator;
pub mod instruction;

use rand::Rng;
use instruction::Instruction;

// Type alias for register
type RegisterIndex = u8;

/// Struct defining the parameters to setup the variable and constant registers
#[derive(Clone)]
pub struct RegisterConfig {
    /// Total number of variable registers
    pub total_var_registers: usize,
    /// Total number of constant registers
    pub total_const_registers: usize,
    /// The lower bound on the range of constants
    pub const_start: f64,
    /// The increment step size for producing the constant range
    pub const_step_size: f64,
    /// Index number indicating the input register
    pub input_register: usize,
    /// Index number indicating the output register
    pub output_register: usize,
    /// The number all variable registers are initialized with
    pub initial_var_value: f64
}

/// The struct which defines the Program. It contains 3 members:
/// - `instructions`: This is the sequence of arithmetic instructions which will execute when the program is run:
/// - `var_registers`: These are initialized with a constant value at the begining of each test case and are subsequently used for calculation through the running of a program.
/// - `const_registers`: These hold a range of read-only constants to be pulled into the calculations during the running of the program.
#[derive(Clone)]
pub struct Program {
    /// Vector holding the sequence of Instructions which defines the Program.
    pub instructions: Vec<Instruction>,
    var_registers: Vec<f64>,
    const_registers: Vec<f64>,
    config: RegisterConfig
}

impl Program {
    /// Create a new Program.
    ///
    /// # Arguments
    /// * `initial_size`: The number of instructions the new program will be initialized with.
    /// * `config`: The register configuration parameters.
    #[allow(clippy::cast_precision_loss)]
    #[must_use] pub fn new(initial_size: usize, config: &RegisterConfig) -> Self {
        // Pre-allocate all vectors with exact capacity
        let mut program = Self {
            instructions: Vec::with_capacity(initial_size),
            var_registers: vec![config.initial_var_value; config.total_var_registers],
            const_registers: Vec::with_capacity(config.total_const_registers),
            config: RegisterConfig { // Avoid clone by constructing directly
                total_var_registers: config.total_var_registers,
                total_const_registers: config.total_const_registers,
                const_start: config.const_start,
                const_step_size: config.const_step_size,
                input_register: config.input_register,
                output_register: config.output_register,
                initial_var_value: config.initial_var_value
            }
        };

        // Fill instructions
        program.instructions.extend((0..initial_size).map(|_|
            Instruction::random(config.total_var_registers, config.total_const_registers)
        ));

        // Fill const registers
        program.const_registers.extend((0..config.total_const_registers).map(|i|
            config.const_start + (i as f64) * config.const_step_size
        ));

        //Program::mark_introns(&mut program.instructions, config.output_register);
        Program::effinit(&mut program.instructions, config.output_register);
        program
    }

    // Ensures all instructions in a program are effective. This is achieved 
    // by modifying the destination register of the intron.
    fn effinit(code: &mut [Instruction], output_register: usize) {
        // Mark the introns of the given program
        Program::mark_introns(code, output_register);

        // Get the index of each intron
        let intron_indices: Vec<usize> = code.iter()
            .enumerate()    // This gives us the actual indices
            .filter(|(_, inst)| inst.0 & 0x8000_0000 == 0)  // The introns
            .map(|(idx, _)| idx) // Isolate the index of the introns
            .collect(); // Collect them

        
        // Get all the effective registers
        let mut effective_regs = vec![
            RegisterIndex::try_from(output_register).expect("Failed to cast output register")
        ];

        for inst in code.iter().rev() {
            effective_regs.extend(inst.operands());
        }

        // Remove duplicates
        effective_regs.sort_unstable();
        effective_regs.dedup();

        // Replace each intron's destination register with a
        // randomly selected effective register
        let mut rng = rand::rng();
        let er_len: usize = effective_regs.len();
        for idx in intron_indices {
            // Clear the destination register
            code[idx].0 &= 0xFF00_FFFF;

            // Generate new effective replacement register
            let new_dst_idx: usize = rng.random_range(..er_len);
            let new_dst: u8 = effective_regs[new_dst_idx];
            
            // Replace it with new value
            code[idx].0 |= (new_dst as u32) << 16;
        }

        // Re-mark to verify all instructions run
        Program::mark_introns(code, output_register);
    }

    /// Mark the effective instructions in given code
    ///
    /// # Arguments
    /// - `code`: Full program to be marked
    #[allow(clippy::missing_panics_doc)]
    pub fn mark_introns(code: &mut [Instruction], output_register: usize) {
        let mut effective_regs: Vec<RegisterIndex> = vec![
            RegisterIndex::try_from(output_register).expect("Failed to cast")
        ];

        for inst in code.iter_mut().rev() {
            if effective_regs.contains(&inst.dst()) {
                effective_regs.extend(inst.operands());
                inst.0 |= 0x8000_0000;
            }
        }
    }

    /// Executes the sequences of instructions on given input and returns the final output of the
    /// program, given the input.
    ///
    /// # Arguments
    /// * `input`: Input value which the program will operate on.
    #[allow(clippy::missing_panics_doc)]
    pub fn run(&mut self, input: f64) -> f64 {
        // Reset variable registers before each run
        self.var_registers = vec![
            self.config.initial_var_value;
            self.config.total_var_registers
        ];

        // Set the input register
        self.var_registers[self.config.input_register] = input;

        for inst in &self.instructions {
            // Check the effective instruction flag
            if inst.0 & 0x8000_0000 == 0 {
                continue;
            }

            let mut operands = [0.0; 2];
            for (i, &idx) in inst.operands().iter().enumerate() {
                operands[i] = if idx < u8::try_from(self.config.total_var_registers).expect("Failed to cast to u8") {
                    self.var_registers[idx as usize]
                } else {
                    self.const_registers[(idx as usize) - self.config.total_var_registers]
                };
            }

            self.var_registers[inst.dst() as usize] = inst.operator()
                .execute(operands[0], operands[1]);
        }
        self.var_registers[self.config.output_register]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_new() {
        let initial_size: usize = 12;
        let config = RegisterConfig {
            total_var_registers: 8,
            total_const_registers: 100,
            const_start: -50.0,
            const_step_size: 1.0,
            input_register: 1,
            output_register: 0,
            initial_var_value: 1.0

        };

        let program = Program::new(initial_size, &config);
        assert_eq!(program.instructions.len(), initial_size);
    }

    #[test]
    fn test_program_run() {
        let input: f64 = 3.0;
        let inst1 = Instruction(0x00023901); // VR[2] = CR[49] + VR[1]
        let inst2 = Instruction(0x02033C02); // VR[3] = CR[52] * VR[2]
        let inst3 = Instruction(0x03040302); // VR[4] = VR[3] / VR[2]
        let inst4 = Instruction(0x03053B3C); // VR[5] = CR[51] / CR[52]
        let inst5 = Instruction(0x0100043D); // VR[0] = VR[4] - CR[53]
        let inst6 = Instruction(0x01000005); // VR[0] = VR[0] - VR[5]

        /*
         * VR[2] = -1.0 + 3.0 = 2.0
         * VR[3] = 2.0 * 2.0  = 4.0
         * VR[4] = 4.0 / 2.0  = 2.0
         * VR[5] = 1.0 / 2.0  = 0.5
         * VR[0] = 2.0 - 3.0  = -1.0
         * VR[0] = -1.0 - 0.5 = -1.5
         */

        let config = RegisterConfig {
            total_var_registers: 8,
            total_const_registers: 100,
            const_start: -50.0,
            const_step_size: 1.0,
            input_register: 1,
            output_register: 0,
            initial_var_value: 1.0

        };

        let mut prog = Program::new(6, &config);
        let mut inst_vec: Vec<Instruction> = vec![
            inst1, inst2,
            inst3, inst4,
            inst5, inst6
        ];

        Program::mark_introns(&mut inst_vec, config.output_register);
        prog.instructions = inst_vec;

        assert_eq!(prog.run(input), -1.5);
    }

    //#[test]
    fn test_intron_marking() {
        let inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3] <-- Intron
        let inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3] <-- Intron
        let inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]


        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];
        Program::mark_introns(&mut instructions, 0);

        // We must confirm the correct instructions were marked as introns
        assert!(instructions[0].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[1].0 & 0x8000_0000 == 0);
        assert!(instructions[2].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[3].0 & 0x8000_0000 == 0);
        assert!(instructions[4].0 & 0x8000_0000 == 0x8000_0000);
    }

    //#[test]
    fn test_effective_code_run() {
        let input: f64 = 2.0;
        let inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3]
        let inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3]
        let inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]

        /*
         * VR[2] = 2.0 + 1.0 = 3.0
         * VR[4] = 3.0 * 1.0 = 3.0
         * VR[1] = 1.0 + 2.0 = 3.0
         * VR[5] = 3.0 * 1.0 = 3.0
         * VR[0] = 3.0 + 3.0 = 6.0
         */

        let config = RegisterConfig {
            total_var_registers: 8,
            total_const_registers: 100,
            const_start: -50.0,
            const_step_size: 1.0,
            input_register: 1,
            output_register: 0,
            initial_var_value: 1.0

        };

        // Create the program
        let mut prog = Program::new(5, &config);

        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];

        // Mark introns and update program's code
        Program::mark_introns(&mut instructions, config.output_register);
        prog.instructions = instructions;

        // Confirm the output equals the correct output
        assert_eq!(prog.run(input), 6.0);
    }

    #[test]
    fn test_effinit() {
        // Write test manually creating program
        // and confirming that all introns are replaced
        // with effective registers 
        todo!()
    }
}
