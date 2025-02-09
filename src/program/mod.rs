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

use instruction::Instruction;

// Type alias for register
type RegisterIndex = u8;

/// Constant which defines the total number of variable or 'calculation'
/// registers used by any program. Variable registers can be both read from and written to. Variable registers are initialized with a constant value. Careful consideration should be made when setting this parameter. Especially if input and output registers are low, sufficient registers is important for performance.
pub const TOTAL_VAR_REGISTERS: usize = 8;

/// Constant which defines the number of constant registers used by the program. Constant registers contain values from a specified range. The constant registers are read-only and present an efficient way to allow a range of values to be introduced into the test case.
pub const TOTAL_CONST_REGISTERS: usize = 100;

/// Before any program is run, its variable register at the index defined by this parameter
/// will be initialized with the given input.
pub const INPUT_REGISTER: usize = 1;

/// After a program is run, what is considered the "output" of the program is defined as the value
/// stored at this index in the program's variable registers.
pub const OUTPUT_REGISTER: usize = 0;

/// Value which defines the lower bound on the values which populate the program's constant
/// registers.
pub const CONST_LOWER_BOUND: f64 = -50.0;

/// Value which defines the upper bound on the values which populate the program's constant
/// registers.
pub const CONST_UPPER_BOUND: f64 = 50.0;

/// The struct which defines the Program. It contains 3 members:
/// - `instructions`: This is the sequence of arithmetic instructions which will execute when the program is run:
/// - `var_registers`: These are initialized with a constant value at the begining of each test case and are subsequently used for calculation through the running of a program.
/// - `const_registers`: These hold a range of read-only constants to be pulled into the calculations during the running of the program.
#[derive(Clone)]
pub struct Program {
    /// Vector holding the sequence of Instructions which defines the Program.
    pub instructions: Vec<Instruction>,
    var_registers: [f64; TOTAL_VAR_REGISTERS],
    const_registers: [f64; TOTAL_CONST_REGISTERS],
}

impl Program {
    /// Create a new Program.
    ///
    /// # Arguments
    /// * `initial_size`: The number of instructions the new program will be initialized with.
    #[allow(clippy::cast_precision_loss)]
    #[must_use] pub fn new(initial_size: usize) -> Self {
        let mut instructions: Vec<Instruction> = (0..initial_size)
            .map(|_| Instruction::random())
            .collect();
        
       Program::mark_introns(&mut instructions);

        let mut program = Self {
            instructions,
            var_registers: [1.0; TOTAL_VAR_REGISTERS],
            const_registers: [0.0; TOTAL_CONST_REGISTERS]
        };

        // Equal step range from lower to upper
        let lower: f64 = CONST_LOWER_BOUND;
        let upper: f64 = CONST_UPPER_BOUND;
        let step: f64 = (upper - lower) / (TOTAL_CONST_REGISTERS as f64);
        program.const_registers = std::array::from_fn(|i| lower + (i as f64) * step);

        program
    }

    /// Mark the effective instructions in given code
    ///
    /// # Arguments
    /// - `code`: Full program to be marked
    #[allow(clippy::missing_panics_doc)]
    pub fn mark_introns(code: &mut [Instruction]) {
        let mut effective_regs: Vec<RegisterIndex> = vec![
            RegisterIndex::try_from(OUTPUT_REGISTER).expect("Failed to cast")
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
        self.var_registers = [1.0; TOTAL_VAR_REGISTERS];

        // Set the input register
        self.var_registers[INPUT_REGISTER] = input;

        for inst in &self.instructions {
            // Check the effective instruction flag
            if inst.0 & 0x8000_0000 == 0 {
                continue;
            }

            let mut operands = [0.0; 2];
            for (i, &idx) in inst.operands().iter().enumerate() {
                operands[i] = if idx < u8::try_from(TOTAL_VAR_REGISTERS).expect("Failed to cast to u8") {
                    self.var_registers[idx as usize]
                } else {
                    self.const_registers[(idx as usize) - TOTAL_VAR_REGISTERS]
                };
            }

            self.var_registers[inst.dst() as usize] = inst.operator()
                .execute(operands[0], operands[1]);

            println!("{inst} = {:?}", self.var_registers[inst.dst() as usize]);
        }
        self.var_registers[OUTPUT_REGISTER]
    }

    /// To display the instructions of the program in human-readable form
    pub fn display(&self) {
        for inst in &self.instructions {
            println!("{inst}");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_new() {
        let initial_size: usize = 12;
        let program = Program::new(initial_size);

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

        let mut prog = Program::new(6);
        let mut inst_vec: Vec<Instruction> = vec![
            inst1, inst2, 
            inst3, inst4, 
            inst5, inst6
        ];

        Program::mark_introns(&mut inst_vec);
        prog.instructions = inst_vec;

        assert_eq!(prog.run(input), -1.5);
    }

    #[test]
    fn test_intron_marking() {
        let input: f64 = 2.0; 
        let inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3] <-- Intron
        let inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3] <-- Intron
        let inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]
        
        
        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];
        Program::mark_introns(&mut instructions);

        // We must confirm the correct instructions were marked as introns
        assert!(instructions[0].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[1].0 & 0x8000_0000 == 0);
        assert!(instructions[2].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[3].0 & 0x8000_0000 == 0);
        assert!(instructions[4].0 & 0x8000_0000 == 0x8000_0000);
    }

    #[test]
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

        // Create the program
        let mut prog = Program::new(5);
        
        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];

        // Mark introns and update program's code
        Program::mark_introns(&mut instructions);
        prog.instructions = instructions;

        // Confirm the output equals the correct output
        assert_eq!(prog.run(input), 6.0);
    }
}

