//! Basic arithmetic operations available to evolving programs.
//! 
//! These operators form the basic building blocks from which more complex
//! mathematical functions can be evolved. The operator set is intentionally
//! small to maintain a focused search space while still allowing for the
//! expression of a wide range of mathematical functions.

/// Available arithmetic operations for program instructions.
/// 
/// Each operator performs a basic arithmetic operation on two operands.
/// The operator set is deliberately minimal to maintain a focused search
/// space while still allowing programs to evolve complex mathematical
/// relationships.
use rand::Rng;

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Operator {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3
}

impl Operator {
    const VARIANTS: [Self; 4] = [Self::Add, Self::Sub, Self::Mul, Self::Div];

    pub fn execute(self, a: f64, b: f64) -> f64 {
        const UNDEFINED: f64 = 1e6;
        match self {
            Operator::Add => a + b,
            Operator::Sub => a - b,
            Operator::Mul => a * b,
            Operator::Div => {
                if b == 0.0 {
                    a + UNDEFINED
                } else {
                    a / b
                }
            }
        }
    }

    pub fn random() -> Self {
        let mut rng = rand::rng();
        Self::VARIANTS[rng.random_range(0..Self::VARIANTS.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_standard() {
        let add: Operator = Operator::Add;
        let sub: Operator = Operator::Sub;
        let mul: Operator = Operator::Mul;
        let div: Operator = Operator::Div;

        let op1 = 4.0;
        let op2 = 5.0;

        assert_eq!(add.execute(op1, op2), 9.0);
        assert_eq!(sub.execute(op1, op2), -1.0);
        assert_eq!(mul.execute(op1, op2), 20.0);
        assert_eq!(div.execute(op1, op2), 0.8);
    }

    #[test]
    fn test_operator_edge() {
        let add: Operator = Operator::Add;
        let sub: Operator = Operator::Sub;
        let mul: Operator = Operator::Mul;
        let div: Operator = Operator::Div;

        let op1 = 1.0;
        let op2 = 0.0;

        assert_eq!(add.execute(op1, op2), 1.0);
        assert_eq!(sub.execute(op1, op2), 1.0);
        assert_eq!(mul.execute(op1, op2), 0.0);
        assert_eq!(div.execute(op1, op2), 1000001.0);
    }

    #[test]
    fn test_operator_random() {
        let op: Operator = Operator::random();
        assert!(matches!(op, Operator::Add | Operator::Sub | Operator::Mul | Operator::Div));
    }
}
