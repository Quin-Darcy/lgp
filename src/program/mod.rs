mod operator;
mod instruction;

// Register configuration
type RegisterIndex = u8;
pub const TOTAL_REGISTERS: usize = 8;
pub const INPUT_REGISTER: RegisterIndex = 1;
pub const OUTPUT_REGISTER: RegisterIndex = 0;
