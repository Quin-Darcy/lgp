use crate::program::Program;

/// Calculate the Mean-Squared Error between set of expected values
/// and the output of the given program for the given set of inputs
///
/// # Arguments
/// - `p`: The `Program` to be evaluated
/// - `training_data`: Set of data containing inputs and expected outputs
#[allow(clippy::cast_precision_loss)]
pub fn mse(p: &mut Program, training_data: &[(f64, f64)]) -> f64 {
    if training_data.is_empty() {
        return 0.0;
    }

    let len = training_data.len() as f64;
    training_data.iter()
        .map(|(input, expected)| {
            let diff = p.run(*input) - expected;
            diff * diff
        })
        .sum::<f64>() / len
}

/// Return the index of the smallest element in an vector slice
///
/// # Arguments
/// - `elements`: Set of elements we are searching
pub fn smallest_element_index(elements: &[f64]) -> usize {
    let mut smallest_element = f64::MAX;
    let mut index: usize = 0;

    for (i, e) in elements.iter().enumerate() {
        if *e < smallest_element {
            smallest_element = *e;
            index = i;
        }
    }
    index
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::instruction::Instruction;

    #[test]
    fn test_utility_mse() {
        let training_data: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 4.0),
            (3.0, 9.0),
            (4.0, 16.0)
        ];

        // Create the program
        let mut prog = Program::new(3);

        // Create vector with instructions, then send it to have effective code marked
        let mut instructions: Vec<Instruction> = vec![
            Instruction(0x0002_0401),   // VR[2] = VR[4] + VR[1]
            Instruction(0x0103_0501),   // VR[3] = VR[5] - VR[1]
            Instruction(0x0200_0203)    // VR[0] = VR[2] * VR[3]
        ];
        Program::mark_introns(&mut instructions);

        // Set the program's instructions equal to the marked code
        prog.instructions = instructions;


        /*
         *  VR[0] = VR[2] * VR[3]
         *        = (VR[4] + VR[1]) * (VR[5] - VR[1])
         *        = (1.0 + VR[1]) * (1.0 - VR[1])
         *        = 1.0 - VR[1] + VR[1] - (VR[1])^2
         *        = 1.0 - (VR[1])^2
         *
         *  ----------------------------------------
         *
         *  p(x) = 1.0 - x^2
         *
         *  p(0.0) = 1.0
         *  p(1.0) = 0.0
         *  p(2.0) = -3.0
         *  p(3.0) = -8.0
         *  p(4.0) = -15.0
         *
         *  (1.0 - 0.0)^2       = 1.0
         *  (0.0 - 1.0)^2       = 1.0
         *  (-3.0 - 4.0)^2      = 49.0
         *  (-8.0 - 9.0)^2      = 289.0
         *  (-15.0 - 16.0)^2    = 961.0
         *
         *  1.0 + 1.0 + 49.0 + 289.0 + 961.0 = 1301.0
         *
         *  1301.0 / 5.0 = 260.2
         */

        assert_eq!(mse(&mut prog, &training_data), 260.2);
    }

    #[test]
    fn test_utility_smallest_element_index() {
        let elements: Vec<f64> = vec![
            45.6,
            28.9,
            2.04,
            3.11,
            5.67
        ];

        assert_eq!(smallest_element_index(&elements), 2);
    }
}
