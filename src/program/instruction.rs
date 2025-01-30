use rand::Rng;

use crate::program::operator::Operator;
use crate::program::{RegisterIndex, TOTAL_REGISTERS};

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction(u32);

impl Instruction {
    pub fn new(
        op: Operator,
        dst: RegisterIndex,
        opnd1: RegisterIndex,
        opnd2: RegisterIndex
    ) -> Self {
        let packed = (op as u32) << 24 |
            (dst as u32) << 16 |
            (opnd1 as u32) << 8 |
            (opnd2 as u32);

        Self(packed)
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();

        let operator: Operator = Operator::random();
        let dst_reg_index: RegisterIndex = rng.random_range(..TOTAL_REGISTERS) as RegisterIndex;
        let opnd1_reg_index: RegisterIndex = rng.random_range(..TOTAL_REGISTERS) as RegisterIndex;
        let opnd2_reg_index: RegisterIndex = rng.random_range(..TOTAL_REGISTERS) as RegisterIndex;

        Self::new(operator, dst_reg_index, opnd1_reg_index, opnd2_reg_index)
    }

    pub fn operator(&self) -> Operator {
        let op_num: u8 = (self.0 >> 24) as u8;
        unsafe { std::mem::transmute::<u8, Operator>(op_num) }
    }

    pub fn dst(&self) -> RegisterIndex {
        ((self.0 & 0x00FF0000) >> 16) as RegisterIndex

    }

    pub fn operands(&self) -> [RegisterIndex; 2] {
        let opnd1: RegisterIndex = ((self.0 & 0x0000FF00) >> 8) as RegisterIndex;
        let opnd2: RegisterIndex = (self.0 & 0x000000FF) as RegisterIndex;
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
        assert!(dst < (TOTAL_REGISTERS as u8));

        let opnds: [RegisterIndex; 2] = inst.operands();
        assert!(opnds[0] < (TOTAL_REGISTERS as u8));
        assert!(opnds[1] < (TOTAL_REGISTERS as u8));
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
