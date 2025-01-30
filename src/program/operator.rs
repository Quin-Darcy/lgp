#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Operator {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3
}

impl Operator {
    pub fn execute(self, a: f64, b: f64) -> f64 {
        const UNDEFINED: f64 = 1e6;
        match self {
            Operator::Add => a + b,
            Operator::Sub => a - b,
            Operator::Mul => a * b,
            Operator::Div => {
                // Protected division
                if b != 0.0 {
                    a / b
                } else {
                    a + UNDEFINED
                }
            },
       } 
    }
}
