pub mod program;        // TODO: Make private after testing

use program::Program;

pub struct LGP {
    pub program: Program
}

impl LGP {
    pub fn new() -> Self {
        Self {
            program: Program::new(),
        }
    } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisrt_test() {
        todo!();
    }
}
