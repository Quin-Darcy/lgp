mod operator;
mod instruction;

pub use instruction::Instruction;
pub use operator::Operator;

// Register configuration
pub type RegisterIndex = u8;
pub const TOTAL_REGISTERS: usize = 8;
pub const INPUT_REGISTER: RegisterIndex = 1;
pub const OUTPUT_REGISTER: RegisterIndex = 0;

pub struct Program {
    instructions: Vec<Instruction>,
    pub registers: [f64; TOTAL_REGISTERS],
}

impl Program {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            registers: [1.0; TOTAL_REGISTERS],
        }
    }
}

