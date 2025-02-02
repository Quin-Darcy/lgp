use crate::program::Program;

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

    #[test]
    fn test_fitness_mse() {
        todo!()
    }
}
