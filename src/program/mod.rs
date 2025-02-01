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
// since RegisterIndex is defined as a u8 with a max value of 256

pub struct Program {
    instructions: Vec<Instruction>,
    var_registers: [f64; TOTAL_VAR_REGISTERS],
    const_registers: [f64; TOTAL_CONST_REGISTERS],
}

impl Program {
    #[allow(clippy::cast_precision_loss)]
    pub fn new(initial_size: usize) -> Self {
        let mut program = Self {
            instructions: Vec::new(),
            var_registers: [1.0; TOTAL_VAR_REGISTERS],
            const_registers: [0.0; TOTAL_CONST_REGISTERS]
        };

        program.instructions = std::iter::repeat_with(Instruction::random)
            .take(initial_size)
            .collect();

        // Equal step range from lower to upper
        let lower: f64 = CONST_LOWER_BOUND;
        let upper: f64 = CONST_UPPER_BOUND;
        let step: f64 = (upper - lower) / (TOTAL_CONST_REGISTERS as f64);
        program.const_registers = std::array::from_fn(|i| lower + (i as f64) * step);

        /*
        // Random values picked from range
        program.const_registers.iter_mut()
            .zip(CONST_LOWER_BOUND as i32..=CONST_UPPER_BOUND as i32)
            .for_each(|(reg, val)| *reg = val as f64);

        */
        program
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn run(&mut self, input: f64) -> f64 {
        self.var_registers[INPUT_REGISTER] = input;
        for inst in &self.instructions {
            let mut operands = [0.0; 2];
            for (i, &idx) in inst.operands().iter().enumerate() {
                operands[i] = if idx < u8::try_from(TOTAL_VAR_REGISTERS).expect("Failed to cast to u8") {
                    self.var_registers[idx as usize]
                } else {
                    self.const_registers[(idx as usize) - TOTAL_VAR_REGISTERS]
                };
            }

            /*
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
            */
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

    #[test]
    fn test_program_run() {
        let input: f64 = 3.0;
        let inst1 = Instruction(0x00023901);
        let inst2 = Instruction(0x02033C02);
        let inst3 = Instruction(0x03040302);
        let inst4 = Instruction(0x03053B3C);
        let inst5 = Instruction(0x0100043D);
        let inst6 = Instruction(0x01000005);

        let inst_vec: Vec<Instruction> = vec![inst1, inst2, inst3, inst4, inst5, inst6];

        let mut prog = Program::new(6);
        prog.instructions = inst_vec;

        assert_eq!(prog.run(input), -1.5);
    }
}
