//! Structure containing the population and methods for evolving the population.
//!
//! This module provides the main data structure whose methods implement the genetic algorithm
//! which evolves the programs. An instance of a Population consists of the following fields:
//!
//! - `individuals`: This is a vector of Programs. 
//! - `fitnesses`: Vector containing the fitness value for each individual.
//!
//! This module defines how individual programs in the genetic programming system
//! are represented and executed. Each program consists of:
//!
//! - A sequence of register-based instructions
//! - Variable registers for calculations
//! - Constant registers with fixed values
//!
//! Programs take a single input and produce a single output, making them
//! suitable for evolving mathematical functions.
//!
//! # Example
//!
//! ```
//! use lgp::population::Population;
//! use lgp::program::Program;
//!
//! // Create a population with initial size 20
//! let mut population = Population::new(20);
//!
//! // Evolve a program against a given training set
//! let program: Program = population.evolve(training_data);
//! ```

mod fitness; 

use crate::program::Program;
use fitness::mse;
use rand::Rng;

/// This is the initial size used for each Program in the set of individuals.
const PROGRAM_SIZE: usize = 20; 

/// Sets the size of the tournament groups. The bigger it is relative to population size, the greater the selection pressure but more pressure means less diversity.
const TOURNAMENT_SIZE: usize = 5;

struct TournamentResult {
    winners: [usize; 2],
    losers: [usize; 2]
}

/// Structure containing the population of Programs.
pub struct Population {
    /// Vector containing the population of Programs.
    pub individuals: Vec<Program>,
    fitnesses: Vec<f64>
}

impl Population {
    /// Create new population of Programs.
    ///
    /// # Arguments
    /// * `population_size`: The number of individuals in the population.
    #[must_use]
    pub fn new(population_size: usize) -> Self {
        let mut indvs: Vec<Program> = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            indvs.push(Program::new(PROGRAM_SIZE));
        }

        Self {
            individuals: indvs,
            fitnesses: vec![0.0; population_size]
        }
    }

    /// Main genetic algorithm loop to evolve a program against the given training data.
    ///
    /// # Arguments
    /// * `training_data`: Set of data points against which the population's fitness is measured.
    pub fn evolve(&mut self, training_data: &[(f64, f64)]) -> Program {
       /*
        * The main loop will consist of the following steps:
        * 1. Evaluate fitness of each individual.
        * 2. Perform selection.
        * 3. Probabilistically perform mutation and crossover.
        * 4. Fill population with new individuals from last step.
        * 5. Repeat until individual fitness high enough.
        */

        self.eval_fitness(training_data);
        todo!()
    }

    /// Computes the fitness value for each program and stores it.
    ///
    /// # Arguments
    /// * `training_data`: Data points against which fitness is measured
    pub fn eval_fitness(&mut self, training_data: &[(f64, f64)]) {
        
        for i in 0..training_data.len() {
            self.fitnesses[i] = mse(&mut self.individuals[i], training_data);
        }
    }

    fn tournament_selection(&mut self) -> TournamentResult {
        // Get 2 * TOURNAMENT_SIZE many random individuals from population
        let random_indices: Vec<usize> = Population::select_no_replacement(
            self.individuals.len(),
            2 * TOURNAMENT_SIZE
        );

        // Split the group into halves
        let first_group = &random_indices[..TOURNAMENT_SIZE];
        let second_group = &random_indices[..TOURNAMENT_SIZE];

        todo!()
        
    }

    // Return highest (lowest) fitness among given group
    fn compete(&self, indices: &[usize]) -> (usize, usize) {
        let mut current_best = f64::MAX;
        let mut current_worst = f64::MIN;
        let mut winner: usize = 0;
        let mut loser: usize = 0;
        for i in indices {
            if self.fitnesses[*i] < current_best {
                current_best = self.fitnesses[*i];
                winner = *i;
            }

            if self.fitnesses[*i] > current_worst {
                current_worst = self.fitnesses[*i];
                loser = *i;
            }
        }
        (winner, loser)
    }

    // Selects k indices from n of them, without replacement
    fn select_no_replacement(n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "Cannot select more items than available.");
        assert!(k != 0, "k must be greater than 0.");

        let mut rng = rand::rng();

        // Create vector of all indices
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        indices.extend(0..n);

        for i in 0..k {
            let j = rng.random_range(i..n);
            indices.swap(i, j);
        }

        indices.truncate(k);
        indices
   } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_new() {
        let pop_size: usize = 38;
        let pop = Population::new(pop_size);
        assert_eq!(pop.individuals.len(), pop_size);
    }
}
