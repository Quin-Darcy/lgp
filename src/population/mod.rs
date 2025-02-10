//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The `Population` data structure 
//! represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::Program;
use rand::Rng;
use std::fmt;

mod utilities;
use utilities::{
    mse, 
    smallest_element_index,
    largest_element_index,
    select_no_replacement
}; 


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

// For each `Program` randomly generated, it will have anywhere
// from 2 to `MAX_INIT_PROG_SIZE` many instructions
const MAX_INIT_PROG_SIZE: usize = 15;

// This parameter sets how many `Programs` compete in a tournament.
// The higher the number, the greater the selection pressure but 
// this decreases diversity.
const TOURNAMENT_SIZE: usize = 4;
struct TournamentResult {
    winners: [usize; 2],
    losers: [usize; 2]
}

impl Population {
    /// Create a new Population and initialize the fitness values
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
        let pop_size = if population_size % 2 == 0 { 
            population_size 
        } else {
            population_size + 1
        };
        let mut programs: Vec<Program> = (0..pop_size)
            .map(|_| Program::new(rng.random_range(2_usize..MAX_INIT_PROG_SIZE)))
            .collect();

        // Collect the fitness values of the random programs
        let fitness_values: Vec<f64> = programs
            .iter_mut()
            .map(|x| mse(x, &training_data))
            .collect();

        // The best is defined as the program with the smallest fitness value
        let training_best_index = smallest_element_index(&fitness_values);

        Self {
            programs,
            fitness_values,
            training_best_index,
            validation_best_index: 0,
            training_data,
            validation_data
        }
    }

    /// Main genetic algorithm loop to evolve a program against the training data.
    pub fn evolve(&mut self) -> Program {
        /*
         * The main loop consists of the following steps:
         * 1. Peform selectoin to return two winners and two losers.
         * 2. Copy winners.
         * 3. Mutate or crossover winners and add to population.
         * 4. Probabilistically replace losers with original winners.
         * 5. Update training and validation bests.
         */

        // Two old programs will be replaced every generation with
        // two new programs. Therefore after the number of generations
        // elapsed is equal to have the population size, all original
        // programs in the population will have been replaced.
        let max_generations = self.programs.len() / 2;

        for _ in 0..max_generations {
            let selection_results = self.tournament_selection();
        }

        todo!()
    }

    fn tournament_selection(&mut self) -> TournamentResult {
        // Get 2 * TOURNAMENT_SIZE many random individuals
        let competitors: Vec<usize> = select_no_replacement(
            self.programs.len(),
            2 * TOURNAMENT_SIZE
        );

        todo!()
    }
}

impl fmt::Display for Population {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prog_lens: Vec<usize> = self.programs
            .iter()
            .map(|x| x.instructions.len())
            .collect();

        let avg_prog_len: f64 = prog_lens
            .iter()
            .sum::<usize>() as f64 / (self.programs.len() as f64);

        let avg_fitness_value: f64 = self.fitness_values
            .iter()
            .sum::<f64>() / (self.fitness_values.len() as f64);

        writeln!(f, "--------------------------------------")?;
        writeln!(f, "Average Fitness Value: {avg_fitness_value:.2}")?;
        writeln!(f, "Average Program Length: {avg_prog_len:.1}")?;
        writeln!(f, "Best Training Fitness: {}", self.fitness_values[self.training_best_index])?;
        writeln!(f, "Best Training Length: {}", self.programs[self.training_best_index].instructions.len())?;
        writeln!(f, "Best Validation Fitness: {}", self.fitness_values[self.validation_best_index])?;
        write!(f, "Best Validation Length: {}", self.programs[self.validation_best_index].instructions.len())?;
        write!(f, "\n--------------------------------------")
    }
}
