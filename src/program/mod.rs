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

use std::cmp;
use rand::Rng;
use instruction::Instruction;

// Type alias for register
type RegisterIndex = u8;

/// Struct defining the parameters to setup the variable and constant registers
#[derive(Clone)]
pub struct ProgramConfig {
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
    pub initial_var_value: f64,
    /// Maximum segment length
    pub max_seg_len: usize,
    /// Maximum distance between crossover points
    pub max_cp_dist: usize,
    /// Maximum difference in segment lengths
    pub max_seg_diff: usize,
    /// Number of instructions that can be mutated in a single variation
    pub mutation_step_size: usize,
    /// Minimum program length
    pub min_prog_len: usize,
    /// Maximum program length
    pub max_prog_len: usize
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
    config: ProgramConfig
}

impl Program {
    /// Create a new Program.
    ///
    /// # Arguments
    /// * `initial_size`: The number of instructions the new program will be initialized with.
    /// * `config`: The register configuration parameters.
    #[allow(clippy::cast_precision_loss)]
    #[must_use] pub fn new(initial_size: usize, config: &ProgramConfig) -> Self {
        // Pre-allocate all vectors with exact capacity
        let mut program = Self {
            instructions: Vec::with_capacity(initial_size),
            var_registers: vec![config.initial_var_value; config.total_var_registers],
            const_registers: Vec::with_capacity(config.total_const_registers),
            config: ProgramConfig { // Avoid clone by constructing directly
                total_var_registers: config.total_var_registers,
                total_const_registers: config.total_const_registers,
                const_start: config.const_start,
                const_step_size: config.const_step_size,
                input_register: config.input_register,
                output_register: config.output_register,
                initial_var_value: config.initial_var_value,
                max_seg_len: config.max_seg_len,
                max_cp_dist: config.max_cp_dist,
                max_seg_diff: config.max_seg_diff,
                mutation_step_size: config.mutation_step_size,
                min_prog_len: config.min_prog_len,
                max_prog_len: config.max_prog_len
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
        
        // NOTE: Effective registers are relative to posistion
        // If you are the second to last instruction, then you
        // can only be effective if your destination register 
        // is the output register or coincides with one of the 
        // operands in the last instruction.
        
        // Loop through the introns
        for idx in intron_indices {

            // Initialize effective registers
            let mut eff_regs = vec![
                RegisterIndex::try_from(output_register).expect("failed to cast")
            ];

            // Step through the code backwards and collect
            // all effective registers up to the current
            // instruction at index idx
            for i in (idx..code.len()).rev() {
                if eff_regs.contains(&code[i].dst()) {
                    eff_regs.extend(code[i].operands());
                }
            }

            // Remove duplicates
            eff_regs.sort_unstable();
            eff_regs.dedup();

            // Replace intron with random effective register
            let mut rng = rand::rng();
            let new_dst_idx: usize = rng.random_range(..eff_regs.len());
            let new_dst: u8 = eff_regs[new_dst_idx];
        
            // Clear the destination register
            code[idx].0 &= 0xFF00_FFFF;

            // Replace it with new effective register
            code[idx].0 |= u32::from(new_dst) << 16;
        }

        // Remark introns
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

    /// Performs crossover between this instance and given instance
    /// of `Program`
    ///
    /// # Arguments
    /// - `code`: &[Instruction]
    pub fn crossover(&self, code: &[Instruction]) -> [Program; 2] {
        // Store the lengths of each parent program
        let prog1_len: usize = self.instructions.len();
        let prog2_len: usize = code.len();

        // Order the program lengths
        let smaller_len;
        let larger_len;
        if prog1_len < prog2_len {
            smaller_len = prog1_len;
            larger_len = prog2_len;
        } else {
            smaller_len = prog2_len;
            larger_len = prog1_len;
        }

        // Note that exchanging segments has the potential to alter a
        // program's size by, at most, max_seg_diff. We need to assure
        // that if the program's size is reduced, it doesn't fall below
        // min_prog_len. Similarly, if the program size is increased, it
        // must not exceed max_prog_len.
        //
        // Both these cases can be avoided by accounting for 'how far'
        // the current program sizes are away from the min and max.
        // With that, we can make sure to adjust max_seg_diff so that
        // swapping segments does not result in crossing either
        // boundary.
        //
        // Ex. If prog1 has 10 instructions and the max_prog_len is 12
        // then we must make sure the max_seg_len <= 2. Similarly, if
        // min_prog_len is 9, then max_seg_len <= 1.
        //
        // We see if the program is closer in length to the max or the min.
        // We take the minimum between these two distances. We do this for
        // both programs. This gives us the largest delta either program
        // can change by while assuring neither will fall below or go over
        // either boundaries. We finally take the min between this delta
        // and the max_seg_diff to give us the final allowed maximum
        // segment difference.

        let dist_from_min1: usize = smaller_len - self.config.min_prog_len;
        let dist_from_max1: usize = self.config.max_prog_len - smaller_len;
        let min1: usize = cmp::min(dist_from_min1, dist_from_max1);

        let dist_from_min2: usize = larger_len - self.config.min_prog_len;
        let dist_from_max2: usize = self.config.max_prog_len - larger_len;
        let min2: usize = cmp::min(dist_from_min2, dist_from_max2);

        // Changing the size of either program by this much or less is 
        // totall safe
        let min_dist: usize = cmp::min(min1, min2);

        // Incorporate our parameter should it be smaller
        let max_seg_diff: usize = cmp::min(min_dist, self.config.max_seg_diff);

        // Now we actually begin our choices
        let mut rng = rand::rng();

        // Select first crossover point from smaller program
        let cp1: usize = rng.random_range(0..smaller_len - 1);

        // Select second crossover point from the second program
        // such that it remains in program bounds and the difference
        // between itself and cp1 does not exceed max_cp_dist
        let lower_cp: usize = cmp::max(0, cp1 - self.config.max_cp_dist);

        // Subtract 2 to make sure segment length of at least 1 is possible
        let upper_cp: usize = cmp::min(larger_len - 2, cp1 + self.config.max_cp_dist);
        let cp2: usize = rng.random_range(lower_cp..=upper_cp);

        // Calculate the remaining lengths between each crossover
        // point and the end of the program, for each program
        let remainder1: usize = smaller_len - cp1;
        let remainder2: usize = larger_len - cp2;

        // The minimum between both remainders gives an upper bound
        // which assures we don't generate segments which exceed either
        // program length
        let min_remainder: usize = cmp::min(remainder1, remainder2);

        // Select random first segment length
        let seg_len1: usize = rng.random_range(1..=min_remainder);

        // Select second segment length such that its difference
        // is less than or equal to max_seg_diff
        let lower_seg: usize = cmp::max(1, seg_len1 - max_seg_diff);
        let upper_seg: usize = cmp::min(larger_len, seg_len1 + max_seg_diff);
        let seg_len2: usize = rng.random_range(lower_seg..=upper_seg);

        // Initialize the two new vectors
        let new_prog1_len: usize = smaller_len - seg_len1 + seg_len2;
        let mut new_instructions1: Vec<Instruction> = Vec::with_capacity(
            new_prog1_len
        );

        let new_prog2_len: usize = larger_len - seg_len2 + seg_len1;
        let mut new_instructions2: Vec<Instruction> = Vec::with_capacity(
            new_prog2_len
        );

        // Create first new vector
        new_instructions1.extend_from_slice(
            &self.instructions[..cp1]
        );
        new_instructions1.extend_from_slice(
            &code[cp1..cp1 + seg_len2]
        );

        // If segment ends before end of vector, then you need to extend
        // the remaining part of the vector
        if cp1 + seg_len1 < smaller_len - 1 {
            new_instructions1.extend_from_slice(
                &self.instructions[cp1 + seg_len2..]
            );
        }

        // Create second new vector
        new_instructions2.extend_from_slice(
            &code[..cp2]
        );
        new_instructions2.extend_from_slice(
            &self.instructions[cp2..cp2 + seg_len1]
        );

        if cp2 + seg_len2 < larger_len - 1 {
            new_instructions2.extend_from_slice(
                &code[cp2 + seg_len1..]
            )
        }

        // Create the two new programs
        let mut new_prog1 = Program::new(new_prog1_len, &self.config);
        let mut new_prog2 = Program::new(new_prog2_len, &self.config);

        // Overwrite instructions with recombined instructions
        new_prog1.instructions = new_instructions1;
        new_prog2.instructions = new_instructions2;

        [new_prog1, new_prog2]
    }

    /*
     * Select up to mutation_step_size many instructions.
     * For each of them, make a choise between a micro or macro
     * mutation. Where micro refers to changing a constant or 
     * register. Macro refers to replacing the entire instruction
     * with a random one, deleting an instruction, or adding an 
     * instruction.
     */
    /// Performs mutation on this instance
    pub fn mutate(&self) -> Program {
        todo!()
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

    #[test]
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
        let inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3] <-- Intron
        let inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3] <-- Intron
        let inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]

        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];

        // Mark the introns first and confirm there are non-effective registers
        Program::mark_introns(&mut instructions, 0);

        // We must confirm the correct instructions were marked as introns
        assert!(instructions[0].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[1].0 & 0x8000_0000 == 0);
        assert!(instructions[2].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[3].0 & 0x8000_0000 == 0);
        assert!(instructions[4].0 & 0x8000_0000 == 0x8000_0000);

        // Replace non-effective code with effective code
        Program::effinit(&mut instructions, 0);

        println!("After effinit:");
        println!("0x{:x}", instructions[0].0);
        println!("0x{:x}", instructions[1].0);
        println!("0x{:x}", instructions[2].0);
        println!("0x{:x}", instructions[3].0);
        println!("0x{:x}", instructions[4].0);

        // We must confirm the introns were replaced
        assert!(instructions[0].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[1].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[2].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[3].0 & 0x8000_0000 == 0x8000_0000);
        assert!(instructions[4].0 & 0x8000_0000 == 0x8000_0000);
    }
}
