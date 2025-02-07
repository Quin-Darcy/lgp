//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The `Population` data structure 
//! represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::Program;


/// Main structure for the management and evolution of the programs.
///
/// The members of a `Population` instance are:
/// - `programs`: Vector containing the `Program`s undergoing evolution.
/// - `fitness_values`: The fitness value for each of the programs.
/// - `training_best`: Index to the program that has performed best on the given training data.
/// - `validation_best`: Index to the program that has performed best on unknown data.
/// - `training_data`: Contains the inputs and expected outputs. This is the data against on which
/// the population will be trained.
/// - `validation_data`: This data is used to assess how well a program generalizes outside of the
/// training data.
pub struct Population {
    programs: Vec<Program>,
    fitness_values: Vec<f64>,
    training_best: usize,
    validation_best: usize,
    training_data: Vec<(f64, f64)>,
    validation_data: Vec<(f64, f64)>
}

const INITIAL_PROGRAM_SIZE: usize = 15;

impl Population {
    /// Create a new Population
    ///
    /// # Arguments
    /// - `population_size`: Number of programs in population
    /// - `training_data`: Data on which to train population
    /// - `validation_data`: Data to test generalizability
    pub fn new(
        population_size: usize, 
        training_data: Vec<(f64, f64)>,
        validation_data: Vec<(f64, f64)>
    ) -> Self {
        // Initialize population as set of random Programs
        let programs: Vec<Program> = (0..population_size)
            .map(|_| Program::new(INITIAL_PROGRAM_SIZE))
            .collect();

        Self {
            programs,
            fitness_values: vec![f64::MAX; population_size],
            training_best: 0,
            validation_best: 0,
            training_data,
            validation_data
        }
    }
}
