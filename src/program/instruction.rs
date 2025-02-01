//! Instructions that form the genetic material of programs.
//! 
//! Instructions are the basic units that can be modified through genetic operations
//! (mutation and crossover). Each instruction is encoded as a 32-bit value that 
//! specifies an operation and the registers it operates on.

/// A single instruction in a program's instruction sequence.
/// 
/// Instructions are the basic units that can be modified through genetic operations.
/// Each instruction performs one arithmetic operation, taking values from two source
/// registers and storing the result in a destination register.
/// 
/// # Bit Layout
/// 
/// The instruction is packed into a 32-bit integer with the following layout:
/// - Bits 24-31: Operator (8 bits)
/// - Bits 16-23: Destination register index (8 bits)
/// - Bits 8-15:  First operand register index (8 bits)
/// - Bits 0-7:   Second operand register index (8 bits)
/// 
/// This compact representation allows for efficient storage and manipulation during
/// genetic operations.
/// 
/// # Examples
/// 
/// ```
/// # use lgp::program::{instruction::Instruction, operator::Operator};
/// // Create a multiplication instruction: dst[3] = reg[4] * reg[5]
/// let inst = Instruction::new(Operator::Mul, 3, 4, 5);
/// 
/// // Instructions can be randomly generated for initial population
/// let random_inst = Instruction::random();
/// ```

use rand::Rng;

use crate::program::operator::Operator;
use crate::program::{RegisterIndex, TOTAL_VAR_REGISTERS, TOTAL_CONST_REGISTERS};

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction(pub u32);

impl Instruction {
    pub fn new(
        op: Operator,
        dst: RegisterIndex,
        opnd1: RegisterIndex,
        opnd2: RegisterIndex
    ) -> Self {
        let packed = (op as u32) << 24 |
            u32::from(dst) << 16 |
            u32::from(opnd1) << 8 |
            u32::from(opnd2);

        Self(packed)
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();

        let operator: Operator = Operator::random();
        let dst_reg_index: RegisterIndex = RegisterIndex::try_from(rng.random_range(..TOTAL_VAR_REGISTERS)).expect("Failed to cast to RegisterIndex");

        // If the reg index > TOTAL_VAR_REGISTERS, it is a const reg index. After this
        // determination, offset by subtracting off TOTAL_VAR_REGISERS
        let opnd1_reg_index: RegisterIndex = RegisterIndex::try_from(rng.random_range(..(TOTAL_VAR_REGISTERS + TOTAL_CONST_REGISTERS))).expect("Failed to cast to RegisterIndex");
        let opnd2_reg_index: RegisterIndex = RegisterIndex::try_from(rng.random_range(..(TOTAL_VAR_REGISTERS + TOTAL_CONST_REGISTERS))).expect("Failed to cast to RegisterIndex");

        Self::new(operator, dst_reg_index, opnd1_reg_index, opnd2_reg_index)
    }

    pub fn operator(&self) -> Operator {
        let op_num: u8 = (self.0 >> 24) as u8;
        unsafe { std::mem::transmute::<u8, Operator>(op_num) }
    }

    pub fn dst(&self) -> RegisterIndex {
        ((self.0 & 0x00FF_0000) >> 16) as RegisterIndex

    }

    pub fn operands(&self) -> [RegisterIndex; 2] {
        let opnd1: RegisterIndex = ((self.0 & 0x0000_FF00) >> 8) as RegisterIndex;
        let opnd2: RegisterIndex = (self.0 & 0x0000_00FF) as RegisterIndex;
        [opnd1, opnd2]
    }
}

// To allow direct comparison with u32
impl PartialEq<u32> for Instruction {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_new() {
        let mul: Operator = Operator::Mul;  // Mul is 2
        let dst: RegisterIndex = 3;
        let opnd1: RegisterIndex = 4;
        let opnd2: RegisterIndex = 5;

        let inst: Instruction = Instruction::new(mul, dst, opnd1, opnd2);

        assert_eq!(inst, 0x02030405);
    }

    #[test]
    fn test_instruction_random() {
        let inst: Instruction = Instruction::random();

        let op = inst.operator();
        assert!(matches!(op, Operator::Add | Operator::Sub | Operator::Mul | Operator::Div));

        let dst: RegisterIndex = inst.dst();
        assert!(dst < (TOTAL_VAR_REGISTERS as u8));

        let opnds: [RegisterIndex; 2] = inst.operands();
        assert!(opnds[0] < ((TOTAL_VAR_REGISTERS + TOTAL_CONST_REGISTERS) as u8));
        assert!(opnds[1] < ((TOTAL_VAR_REGISTERS + TOTAL_CONST_REGISTERS) as u8));
    }

    #[test]
    fn test_instruction_op() {
        let inst: Instruction = Instruction(0x02030405);
        assert_eq!(inst.operator(), Operator::Mul);
    }

    #[test]
    fn test_instruction_dst() {
        let inst: Instruction = Instruction(0x02030405);
        let dst: RegisterIndex = 3;
        assert_eq!(inst.dst(), dst);
    }

    #[test]
    fn test_instruction_opnds() {
        let inst: Instruction = Instruction(0x02030405);
        let opnds: [RegisterIndex; 2] = [4, 5];
        assert_eq!(inst.operands(), opnds);
    }
}
