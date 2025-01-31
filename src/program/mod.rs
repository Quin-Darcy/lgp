mod operator;
mod instruction;

use instruction::Instruction;

// Register configuration
type RegisterIndex = u8;
pub const TOTAL_VAR_REGISTERS: usize = 8;
pub const TOTAL_CONST_REGISTERS: usize = 100;
pub const INPUT_REGISTER: usize = 1;
pub const OUTPUT_REGISTER: usize = 0;
pub const CONST_LOWER_BOUND: f64 = -50.0;
pub const CONST_UPPER_BOUND: f64 = 50.0;

// TODO: Validate that the total number of registers (const and var) are less than 256

pub struct Program {
    instructions: Vec<Instruction>,
    var_registers: [f64; TOTAL_VAR_REGISTERS],
    const_registers: [f64; TOTAL_CONST_REGISTERS],
}

impl Program {
    pub fn new(initial_size: usize) -> Self {
        let mut program = Self {
            instructions: Vec::new(),
            var_registers: [0.0; TOTAL_VAR_REGISTERS],
            const_registers: [0.0; TOTAL_CONST_REGISTERS]
        };

        program.instructions = std::iter::repeat_with(Instruction::random)
            .take(initial_size)
            .collect();

        program.const_registers.iter_mut()
            .zip(CONST_LOWER_BOUND as i32..=CONST_UPPER_BOUND as i32)
            .for_each(|(reg, val)| *reg = val as f64);

        program
    }

    pub fn run(&mut self, input: f64) -> f64 {
        self.var_registers[INPUT_REGISTER] = input;
        for inst in self.instructions.iter() {
            let operands: Vec<f64> = inst.operands()
                .iter()
                .map(|&idx| {
                    if idx < TOTAL_VAR_REGISTERS as u8 {
                        self.var_registers[idx as usize]
                    } else {
                        self.const_registers[(idx as usize) - TOTAL_VAR_REGISTERS]
                    }
                })
                .collect();

                self.var_registers[inst.dst() as usize] = inst.operator().execute(operands[0], operands[1]);
        }
        self.var_registers[OUTPUT_REGISTER]
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
}
