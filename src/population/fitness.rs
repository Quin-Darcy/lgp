use crate::program::Program;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::instruction::Instruction;

    #[test]
    fn test_fitness_mse() {
        let training_data: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 4.0),
            (3.0, 9.0),
            (4.0, 16.0)
        ];
        let mut p = Program::new(3);
        p.instructions = vec![
            Instruction(0x0002_0401),   // vr[2] = vr[4] + vr[1]
            Instruction(0x0103_0501),   // vr[3] = vr[5] - vr[1]
            Instruction(0x0200_0203)    // vr[0] = vr[2] * vr[3]
        ];

        /*
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

        assert_eq!(mse(&mut p, &training_data), 260.2);
    }
}
