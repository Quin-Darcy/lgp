
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
#[allow(clippy::module_name_repetitions)]
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
    /// The rate we insert (versus delete) instruction during macro mutation
    pub insertion_rate: f64,
    /// Minimum program length
    pub min_prog_len: usize,
    /// Maximum program length
    pub max_prog_len: usize
}

// Default config values
impl Default for ProgramConfig {
    fn default() -> Self {
        ProgramConfig {
            total_var_registers:8,
            total_const_registers: 100,
            const_start: -50.0,
            const_step_size: 1.0,
            input_register: 1,
            output_register: 0,
            initial_var_value: 0.0,
            max_seg_len: 5,
            max_cp_dist: 6,
            max_seg_diff: 1,
            mutation_step_size: 1,
            insertion_rate: 0.5,
            min_prog_len: 1,
            max_prog_len: 200
        }
    }
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
                insertion_rate: config.insertion_rate,
                min_prog_len: config.min_prog_len,
                max_prog_len: config.max_prog_len
            }
        };

        // Fill const registers
        program.const_registers
            .extend((0..config.total_const_registers).map(|i|
            config.const_start + (i as f64) * config.const_step_size
        ));

        // Begin by filling the vector with a random instruction
        let inst = Instruction::random(
            config.total_var_registers,
            config.total_const_registers
        );
        for _ in 0..initial_size {
            program.instructions.push(inst);
        }

        // For randomly selecting operand
        let mut rng = rand::rng();

        // TODO: Need to implement vector of effective registers
        // just pulling operands from next does not work since 
        // two operands may be constants

        // Work backwords
        for i in (0..initial_size).rev() {
            // If last instruction, force output register
            // and at least one variable register operand
            if i == initial_size - 1 {
                program.instructions[i].0 &= 0xFF00_FFFF;

                let vr_index: usize = rng.random_range(0..config.total_var_registers);
                // Clear the last operand
                program.instructions[i].0 &= 0xFFFF_FF00;

                // Force the last operand as variable
                program.instructions[i].0 |= u32::try_from(vr_index)
                    .expect("failed to cast");
            } else {
                // Initialize each previous instruction as random
                program.instructions[i] = Instruction::random(
                    config.total_var_registers,
                    config.total_const_registers
                );

                // We make it effective by setting its destination
                // register to be one of the subsequent instruction's
                // operand registers
                let ops = program.instructions[i+1].operands();

                // Select an operand
                let op_index: usize = rng.random_range(0..ops.len());
                let new_dst = u32::try_from(ops[op_index]).expect("cast fail");

                // Clear the destination register
                program.instructions[i].0 &= 0xFF00_FFFF;

                // Set the new destination register
                program.instructions[i].0 |= new_dst << 16;
            }
        }

        program
    }

    // Ensures all instructions in a program are effective. This is achieved 
    // by modifying the destination register of the intron.
    fn effinit(
        code: &mut [Instruction], 
        output_register: usize, 
        total_var_registers: usize
    ) {
        // Leave if code is empty
        if code.is_empty() {
            return;
        }

        // Mark the introns of the given program
        Program::mark_introns(code, output_register, total_var_registers);

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
                    // Update eff_regs by removing destination register
                    eff_regs.retain(|&r| r != code[i].dst());

                    // Only add operands that are variable registers
                    for &op in &code[i].operands() {
                        if op < u8::try_from(total_var_registers)
                            .expect("failed to cast") 
                        {
                            eff_regs.push(op);
                        }
                    }
                }
            }

            // Remove duplicates
            eff_regs.sort_unstable();
            eff_regs.dedup();

            // Only proceed if there are effective registers to choose from
            if !eff_regs.is_empty() {
                // Replace intron with random effective register
                let mut rng = rand::rng();
                let new_dst_idx: usize = rng.random_range(..eff_regs.len());
                let new_dst: u8 = eff_regs[new_dst_idx];
            
                // Clear the destination register
                code[idx].0 &= 0xFF00_FFFF;

                // Replace it with new effective register
                code[idx].0 |= u32::from(new_dst) << 16;
            } else {
                println!("empty shit");
                // If there are no effective registers at this position,
                // make this instruction write to the output register directly
                // This maintains evolutionary integrity while preventing empty 
                // programs
                code[idx].0 &= 0xFF00_FFFF; // Clear destination bits
                code[idx].0 |= u32::from(
                    RegisterIndex::try_from(output_register)
                    .expect("failed to cast")
                ) << 16;
            }
        }

        // Remark introns
        Program::mark_introns(code, output_register, total_var_registers);

        if code.len() == 0 {
            panic!("effinit - zero len");
        }
    }

    /// Mark the effective instructions in given code
    ///
    /// # Arguments
    /// - `code`: Full program to be marked
    #[allow(clippy::missing_panics_doc)]
    pub fn mark_introns(
        code: &mut [Instruction], 
        output_register: usize,
        total_var_regs: usize
    ) {
        let mut effective_regs: Vec<RegisterIndex> = vec![
            RegisterIndex::try_from(output_register).expect("Failed to cast")
        ];

        for inst in code.iter_mut().rev() {
            if effective_regs.contains(&inst.dst()) {
                // Mark the instruction as effective
                inst.0 |= 0x8000_0000;

                // Remove destination register from effective_regs
                effective_regs.retain(|&r| r != inst.dst());

                // Only add operands that are variable registers
                for &op in &inst.operands() {
                    if op < u8::try_from(total_var_regs)
                        .expect("failed to cast") {
                            effective_regs.push(op);
                    }
                }
            } else {
                inst.0 &= 0x0FFF_FFFF;
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

    /// Removes non-effective code
    ///
    /// # Arguments
    /// - `code`: Instructions to reduce
    /// - `output_reg`: Output register
    pub fn remove_introns(
        code: &mut Vec<Instruction>, 
        output_reg: usize,
        total_var_regs: usize
    ) {
        // Create backup 
        let mut code_clone: Vec<Instruction> = code.clone();
        Program::mark_introns(code, output_reg, total_var_regs);
        code.retain(|inst| inst.0 & 0x8000_0000 != 0);

        // If there is no effective code, send it back as it was.
        // (Hopefully) Tournament selection will naturally take care
        // of it
        if code.len() == 0 {
            *code = code_clone;
        }
    }

    /// Performs crossover between this instance and given instance
    /// of `Program`
    ///
    /// # Arguments
    /// - `other_code`: Instructions of other parent which this instance is recombinging with
    #[must_use] pub fn crossover(&self, other_code: &[Instruction]) -> [Program; 2] {
        // Order program lengths
        let (smaller_prog, smaller_len, larger_prog, larger_len): (
            &[Instruction], 
            usize, 
            &[Instruction], 
            usize
        ) = if self.instructions.len() < other_code.len() {
            (
                &self.instructions, 
                self.instructions.len(), 
                other_code, 
                other_code.len()
            )
        } else {
            (
                other_code,
                other_code.len(),
                &self.instructions,
                self.instructions.len()
            )
        };

        let mut rng = rand::rng();

        // Select the first crossover point from the smaller program
        let cp1: usize = rng.random_range(0..=smaller_len.saturating_sub(2));

        // Select second crossover point from the larger program from
        // neighborhood around first crossover point intersect larger
        // program's boundaries
        let lower_cp: usize = cmp::max(0, cp1.saturating_sub(self.config.max_cp_dist));
        let upper_cp: usize = cmp::min(larger_len.saturating_sub(2), cp1 + self.config.max_cp_dist);
        let cp2: usize = rng.random_range(lower_cp..=upper_cp);

        // The first segment length will be selected between 1, 
        // which is the minimum segment length, and the minumum
        // between the largest segment size from the given crossover
        // point and the max_seg_len
        let seg_len_upper1: usize = cmp::min(
            smaller_len.saturating_sub(1 + cp1),
            self.config.max_seg_len
        );
        let seg_len1: usize = if seg_len_upper1 <= 1 {
            seg_len_upper1
        } else {
            rng.random_range(1..=seg_len_upper1) 
        };

        // The second segment needs to satisfy the following constraints
        // - length must be at most max_seg_len
        // - difference in length between seg_len1 is at most max_seg_diff
        // - The delta in the length of either program must be small enough
        // such that neither fall below min_prog_len or exceed max_prog_len
        let dist_to_min: usize = smaller_len.saturating_sub(self.config.min_prog_len);
        let dist_to_max: usize = self.config.max_prog_len.saturating_sub(larger_len);
        let max_len_delta: usize = cmp::min(
            cmp::min(self.config.max_seg_diff, dist_to_min), 
            dist_to_max
        );

        // Calculate the lower and upper bounds on the second segment and select
        let lower_seg_len: usize = cmp::max(1, seg_len1.saturating_sub(max_len_delta));
        let upper_seg_len: usize = cmp::min(larger_len.saturating_sub(1 + cp2), seg_len1 + max_len_delta);
        // Its possible there is less room left from cp2 to end than seg_len1 - max_len_delta
        let seg_len2: usize = if lower_seg_len >= upper_seg_len {
            upper_seg_len
        } else { 
            rng.random_range(lower_seg_len..=upper_seg_len) 
        };

        // Compute the lengths of the new vectors
        let new_prog_len1: usize = smaller_len.saturating_sub(seg_len1) + seg_len2;
        let new_prog_len2: usize = larger_len.saturating_sub(seg_len2) + seg_len1;

        // Create and allocate memory for the new vectors
        let mut new_instructions1: Vec<Instruction> = Vec::with_capacity(
            new_prog_len1
        );
        let mut new_instructions2: Vec<Instruction> = Vec::with_capacity(
            new_prog_len2
        );

        // Fill the new vectors 
        new_instructions1.extend_from_slice(
            &smaller_prog[..cp1]
        );
        if cp2 + seg_len2 < larger_len {
            new_instructions1.extend_from_slice(
                &larger_prog[cp2..cp2 + seg_len2]
            );
        }
        if cp1 + seg_len1 < smaller_len {
            new_instructions1.extend_from_slice(
                &smaller_prog[cp1 + seg_len1..]
            );
        }

        new_instructions2.extend_from_slice(
            &larger_prog[..cp2]
        );
        if cp1 + seg_len1 < smaller_len {
            new_instructions2.extend_from_slice(
                &smaller_prog[cp1..cp1 + seg_len1]
            );
        }
        if cp2 + seg_len2 < larger_len {
            new_instructions2.extend_from_slice(
                &larger_prog[cp2 + seg_len2..]
            );
        }

        // Create the two offspring Programs
        let mut new_prog1 = Program::new(new_prog_len1, &self.config);
        let mut new_prog2 = Program::new(new_prog_len2, &self.config);

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
    #[must_use] pub fn mutate(&self, mutation_type: f64) -> Program {
        let mut rng = rand::rng();

        // Create a mutable clone of the program
        let mut new_prog: Program = self.clone();
        let prog_len: usize = new_prog.instructions.len();

        if rng.random::<f64>() < mutation_type {
            // Macro mutation
            
            let i: usize = rng.random_range(0..prog_len);
            
            // Select between insertion or deletion
            if rng.random::<f64>() < self.config.insertion_rate {
                if prog_len < new_prog.config.max_prog_len ||
                    prog_len == new_prog.config.min_prog_len {
                    // Insertion

                    // Create random instruction
                    let new_inst = Instruction::random(
                        new_prog.config.total_var_registers,
                        new_prog.config.total_const_registers
                    );

                    // Insert the new instruction at the random index
                    new_prog.instructions.insert(i, new_inst);
                    
                    // TODO: Add effective mutation part here
                }
            } else {
                if prog_len > new_prog.config.min_prog_len || 
                    prog_len == new_prog.config.max_prog_len {
                    // Deletion

                    // Delete instruction at the index
                    new_prog.instructions.remove(i);
                }
            }
        } else {
            // Micro mutation
        
        }
        if new_prog.instructions.len() == 0 {
            panic!("Mutate - Zero length program");
        }
        new_prog
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_new() {
        let initial_size: usize = 12;
        let config = ProgramConfig::default();
        let program = Program::new(initial_size, &config);
        assert_eq!(program.instructions.len(), initial_size);
    }

    #[test]
    fn test_program_new_effective() {
        let initial_size: usize = 12;
        let config = ProgramConfig::default();
        let mut program = Program::new(initial_size, &config);

        Program::mark_introns(
            &mut program.instructions,
            config.output_register,
            config.total_var_registers
        );

        for i in 0..program.instructions.len() {
            println!("0x{:08x}", program.instructions[i].0);
        }

        for i in 0..program.instructions.len() {
            assert!(program.instructions[i].0 & 0x8000_0000 == 0x800_0000);
        }
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

        let config = ProgramConfig::default();

        let mut prog = Program::new(6, &config);
        let mut inst_vec: Vec<Instruction> = vec![
            inst1, inst2,
            inst3, inst4,
            inst5, inst6
        ];

        Program::mark_introns(
            &mut inst_vec, 
            config.output_register, 
            config.total_var_registers
        );
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
        Program::mark_introns(&mut instructions, 0, 8);

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

        let config = ProgramConfig {
            total_var_registers:8,
            total_const_registers: 100,
            const_start: -50.0,
            const_step_size: 1.0,
            input_register: 1,
            output_register: 0,
            initial_var_value: 1.0,
            max_seg_len: 5,
            max_cp_dist: 6,
            max_seg_diff: 1,
            mutation_step_size: 1,
            insertion_rate: 0.5,
            min_prog_len: 3,
            max_prog_len: 200
        };

        // Create the program
        let mut prog = Program::new(5, &config);

        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];

        // Mark introns and update program's code
        Program::mark_introns(
            &mut instructions, 
            config.output_register, 
            config.total_var_registers
        );
        prog.instructions = instructions;

        // Confirm the output equals the correct output
        assert_eq!(prog.run(input), 6.0);
    }

    /*
    #[test]
    fn test_effinit() {
        let inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3] <-- Intron
        let inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3] <-- Intron
        let inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]

        // Create vector of all instructions
        let mut instructions = vec![inst1, inst2, inst3, inst4, inst5];

        // Replace non-effective code with effective code
        Program::effinit(&mut instructions, 0, 8);

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
    */

    #[test]
    fn test_remove_introns() {
        let mut inst1 = Instruction(0x0002_0103); // VR[2] = VR[1] + VR[3]
        let inst2 = Instruction(0x0204_0203); // VR[4] = VR[2] * VR[3] <-- Intron
        let mut inst3 = Instruction(0x0001_0301); // VR[1] = VR[3] + VR[1]
        let inst4 = Instruction(0x0205_0203); // VR[5] = VR[2] * VR[3] <-- Intron
        let mut inst5 = Instruction(0x0000_0102); // VR[0] = VR[1] + VR[2]

        // Create vector of all instructions
        let mut instructions = vec![inst1.clone(), inst2, inst3.clone(), inst4, inst5.clone()];

        // Set the high-bit on the effective instructions
        inst1.0 |= 0x8000_0000;
        inst3.0 |= 0x8000_0000;
        inst5.0 |= 0x8000_0000;

        // Create vector containing expected instructions
        let ex_instructions = vec![inst1, inst3, inst5];

        Program::remove_introns(&mut instructions, 0, 8);

        assert_eq!(instructions, ex_instructions);
    }

    #[test]
    fn test_crossover_fuzz() {
        let config = ProgramConfig::default();
        
        // Run many iterations with random program sizes
        let mut rng = rand::rng();
        for _ in 0..100 {
            let len1 = rng.random_range(
                config.min_prog_len..=config.max_prog_len
            );
            let len2 = rng.random_range(
                config.min_prog_len..=config.max_prog_len
            );
            
            let prog1 = Program::new(len1, &config);
            let prog2 = Program::new(len2, &config);
            
            // This should run without panicking
            let offspring = prog1.crossover(&prog2.instructions);
            
            // Basic validation
            assert!(offspring[0].instructions.len() >= config.min_prog_len);
            assert!(offspring[1].instructions.len() >= config.min_prog_len);
            assert!(offspring[0].instructions.len() <= config.max_prog_len);
            assert!(offspring[1].instructions.len() <= config.max_prog_len);
        }
    }

    #[test]
    fn test_crossover_instruction_transfer() {
        let config = ProgramConfig::default();
        
        // Create programs with known instructions
        let mut prog1 = Program::new(0, &config);
        let mut prog2 = Program::new(0, &config);
        
        // Create instructions that are genuinely effective by making all of them
        // manipulate the output register and leaving source registers distinct
        prog1.instructions = (0..10).map(|i| {
            // Format: 0x0000YYZZ where:
            // 00 = operator (0 = Add)
            // 00 = destination register (0 = output register)
            // YY = source register 1 (different for each program)
            // ZZ = source register 2 (increasing with i)
            Instruction(0x00000100 + i)
        }).collect();
        
        prog2.instructions = (0..10).map(|i| {
            // Same pattern but different source register 1
            Instruction(0x00000200 + i)
        }).collect();
        
        // Perform crossover
        let offspring = prog1.crossover(&prog2.instructions);
        
        // Check offspring 1 has instructions from both parents
        let offspring1_has_prog1 = offspring[0].instructions.iter()
            .any(|inst| {
                let src_reg = (inst.0 >> 8) & 0xFF;
                src_reg == 0x01
            });
        
        let offspring1_has_prog2 = offspring[0].instructions.iter()
            .any(|inst| {
                let src_reg = (inst.0 >> 8) & 0xFF;
                src_reg == 0x02
            });
        
        // Check offspring 2 has instructions from both parents
        let offspring2_has_prog1 = offspring[1].instructions.iter()
            .any(|inst| {
                let src_reg = (inst.0 >> 8) & 0xFF;
                src_reg == 0x01
            });
        
        let offspring2_has_prog2 = offspring[1].instructions.iter()
            .any(|inst| {
                let src_reg = (inst.0 >> 8) & 0xFF;
                src_reg == 0x02
            });
        
        assert!(offspring1_has_prog1);
        assert!(offspring1_has_prog2);
        assert!(offspring2_has_prog1);
        assert!(offspring2_has_prog2);
    }
}
