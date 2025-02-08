//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The `Population` data structure 
//! represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::Program;
use rand::Rng;
use std::fmt;


/// Main structure for the management and evolution of the programs.
///
/// The members of a `Population` instance are:
/// - `programs`: Vector containing the `Program`s undergoing evolution.
/// - `fitness_values`: The fitness value for each of the programs.
/// - `training_best_index`: Index to the program that has performed best on the given training data.
/// - `validation_best_index`: Index to the program that has performed best on unknown data.
/// - `training_data`: Contains the inputs and expected outputs. This is the data against on which the population will be trained.
/// - `validation_data`: This data is used to assess how well a program generalizes outside of the training data.
pub struct Population {
    programs: Vec<Program>,
    fitness_values: Vec<f64>,
    training_best_index: usize,
    validation_best_index: usize,
    training_data: Vec<(f64, f64)>,
    validation_data: Vec<(f64, f64)>
}

const MAX_INIT_PROG_SIZE: usize = 15;

impl Population {
    /// Create a new Population
    ///
    /// # Arguments
    /// - `population_size`: Number of programs in population
    /// - `training_data`: Data on which to train population
    /// - `validation_data`: Data to test generalizability
    #[must_use] pub fn new(
        population_size: usize, 
        training_data: Vec<(f64, f64)>,
        validation_data: Vec<(f64, f64)>
    ) -> Self {
        // Initialize population as set of random Programs
        let mut rng = rand::rng();
        let programs: Vec<Program> = (0..population_size)
            .map(|_| Program::new(rng.random_range(2_usize..MAX_INIT_PROG_SIZE)))
            .collect();

        Self {
            programs,
            fitness_values: vec![f64::MAX; population_size],
            training_best_index: 0,
            validation_best_index: 0,
            training_data,
            validation_data
        }
    }
}

impl fmt::Display for Population {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prog_lens: Vec<usize> = self.programs
            .iter()
            .map(|x| x.instructions.len())
            .collect();
        let avg_prog_len: f64 = prog_lens
            .iter()
            .sum::<usize>() as f64 / (self.programs.len() as f64);

        write!(f, "--------------------------------------\n")?;
        write!(f, "Average Program Length: {avg_prog_len}\n")?;
        write!(f, "Best Training Fitness: {}\n", self.fitness_values[self.training_best])?;
        write!(f, "Best Training Length: {}\n", self.programs[self.training_best].instructions.len())?;
        write!(f, "Best Validation Fitness: {}\n", self.fitness_values[self.validation_best])?;
        write!(f, "Best Validation Length: {}", self.programs[self.validation_best].instructions.len())?;
        write!(f, "\n--------------------------------------")
    }
}
