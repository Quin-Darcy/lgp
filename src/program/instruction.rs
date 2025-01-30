use super::RegisterIndex;
use super::Operator;

#[derive(Clone, Copy)]
pub struct Instruction(u32);

impl Instruction {
    pub fn new(
        operator: Operator, 
        dest: RegisterIndex, 
        op1: RegisterIndex, 
        op2: RegisterIndex
    ) -> Self {
       let packed = (operator as u32) << 24 |
           (dest as u32) << 16 |
           (op1 as u32) << 8 |
           (op2 as u32);
        
       Self(packed)
    }
}
