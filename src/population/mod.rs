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

/// Sets the rate at which new offspring are created via crossover.
const CROSSOVER_RATE: f64 = 0.78;

/// Sets the rate at which new offspring are created via mutation.
const MUTATION_RATE: f64 = 1.0 - CROSSOVER_RATE;

/// Sets the rate at which winners are reproduced.
const REPRODUCTION_RATE: f64 = 0.70;

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
        let pop_size = if population_size % 2 == 0 { population_size } else { population_size + 1 };
        let mut indvs: Vec<Program> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            indvs.push(Program::new(PROGRAM_SIZE));
        }

        Self {
            individuals: indvs,
            fitnesses: vec![0.0; pop_size]
        }
    }

    /// Main genetic algorithm loop to evolve a program against the given training data.
    ///
    /// # Arguments
    /// * `training_data`: Set of data points against which the population's fitness is measured.
    /// * `validation_data`: Data points used to measure program's capacity to generalize.
    pub fn evolve(
        &mut self, 
        training_data: &[(f64, f64)],
        validation_data: &[(f64, f64)]
    ) -> Program {

       /*
        * The main loop will consist of the following steps:
        * 1. Evaluate fitness of each individual.
        * -- Loop starts --
        * 2. Perform selection to return two winners and two losers.
        * 3. Copy winners.
        * 4. Mutate or crossover winners and add to population.
        * 5. Probabilistically replace losers with original winners.
        * -- Loop ends after max generations --
        * TODO: Add more info on validation
        */

        self.eval_fitness(training_data);

        let max_generations = self.individuals.len() / 2;

        for _ in 0..max_generations {
            // Returns two winners and two losers
            let results = self.tournament_selection();
            let winner_index1 = results.winners[0];
            let winner_index2 = results.winners[1];

            self.update_population(&results);

            self.fitnesses[winner_index1] = mse(
                &mut self.individuals[winner_index1],
                training_data
            );
            self.fitnesses[winner_index2] = mse(
                &mut self.individuals[winner_index2],
                training_data
            );

            // TODO: Evaluate training fitness of new offspring
            // TODO: Validation new offspring and global best
        }


        todo!()
    }

    fn update_population(&mut self, results: &TournamentResult) {
        // Create copies of winners
        let first_winner = self.individuals[results.winners[0]].clone();
        let second_winner = self.individuals[results.winners[1]].clone();

        let mut rng = rand::rng();
        if rng.random::<f64>() < CROSSOVER_RATE {
            // Get indices first
            let idx1 = results.winners[0];
            let idx2 = results.winners[1];
            
            // Create temporary copies
            let mut prog1 = self.individuals[idx1].clone();
            let mut prog2 = self.individuals[idx2].clone();
            
            // Perform crossover on the copies
            self.crossover(&mut prog1, &mut prog2);
            
            // Update the population with the modified copies
            self.individuals[idx1] = prog1;
            self.individuals[idx2] = prog2;
        } else {
            // Similarly for mutation
            let idx1 = results.winners[0];
            let idx2 = results.winners[1];
            
            let mut prog1 = self.individuals[idx1].clone();
            let mut prog2 = self.individuals[idx2].clone();
            
            self.mutate(&mut prog1);
            self.mutate(&mut prog2);
            
            self.individuals[idx1] = prog1;
            self.individuals[idx2] = prog2;
        }

        if rng.random::<f64>() < REPRODUCTION_RATE {
            self.individuals[results.losers[0]] = first_winner;
            self.individuals[results.losers[1]] = second_winner;
        }
    }

    fn mutate(&mut self, prog: &mut Program) {
        todo!();
    } 

    fn crossover(&mut self, prog1: &mut Program, prog2: &mut Program) {
        /*
         * Plan:
         *
         * Randomly select two indices in both program's instruction vectors.
         * Create a slice which consists of the first segment of instructions
         * up to the first selected point, then a slice consisting of 
         * the segment of the other program's instructions between its selected
         * points
         */

        let mut rng = rand::rng();

        let len1 = prog1.instructions.len();
        let len2 = prog2.instructions.len();

        if len1 < 2 || len2 < 2 {
            return;
        }

        let mut point1_1 = rng.random_range(0..len1);
        let mut point1_2 = rng.random_range(0..len1);
        if point1_1 > point1_2 {
            std::mem::swap(&mut point1_1, &mut point1_2);
        }

        let mut point2_1 = rng.random_range(0..len2);
        let mut point2_2 = rng.random_range(0..len2);
        if point2_1 > point2_2 {
            std::mem::swap(&mut point2_1, &mut point2_2);
        }

        let new_len1 = point1_1 + (point2_2 - point2_1 + 1) + (len1 - point1_2 - 1);
        let new_len2 = point2_1 + (point1_2 - point1_1 + 1) + (len2 - point2_2 - 1);

        let mut new_instructions1 = Vec::with_capacity(new_len1);
        let mut new_instructions2 = Vec::with_capacity(new_len2);

        // Build first program's new instructions
        new_instructions1.extend_from_slice(&prog1.instructions[..point1_1]);
        new_instructions1.extend_from_slice(&prog2.instructions[point2_1..=point2_2]);
        new_instructions1.extend_from_slice(&prog1.instructions[point1_2+1..]);

        // Build second program's new instructions
        new_instructions2.extend_from_slice(&prog2.instructions[..point2_1]);
        new_instructions2.extend_from_slice(&prog1.instructions[point1_1..=point1_2]);
        new_instructions2.extend_from_slice(&prog2.instructions[point2_2+1..]);

        // Replace original instructions with new ones
        prog1.instructions = new_instructions1;
        prog2.instructions = new_instructions2;
    }

    // Computes the fitness value for each program and stores it.
    fn eval_fitness(&mut self, training_data: &[(f64, f64)]) {
        
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

        // Run the two tounrnaments
        let first_results: (usize, usize) = self.compete(first_group);
        let second_results: (usize, usize) = self.compete(second_group);

        TournamentResult {
            winners: [first_results.0, second_results.0],
            losers: [first_results.1, second_results.1]
        }
        
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
