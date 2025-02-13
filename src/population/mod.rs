//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The `Population` data structure 
//! represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::{Program, RegisterConfig};
use crate::program::instruction::Instruction;
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
const MAX_INIT_PROG_SIZE: usize = 5;

// Sets the probability for crossover versus mutation
const CROSSOVER_RATE: f64 = 0.74;

// Sets the probability for performing reproduction
const REPRODUCTION_RATE: f64 = 0.67;

// This parameter sets how many `Programs` compete in a tournament.
// The higher the number, the greater the selection pressure but 
// this decreases diversity.
const TOURNAMENT_SIZE: usize = 20;
#[derive(Clone)]
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
        validation_data: Vec<(f64, f64)>,
        reg_config: RegisterConfig
    ) -> Self {
        // Initialize population as set of random Programs
        let mut rng = rand::rng();
        let pop_size = if population_size % 2 == 0 { 
            population_size 
        } else {
            population_size + 1
        };
        let mut programs: Vec<Program> = (0..pop_size)
            .map(|_| Program::new(
                    rng.random_range(2_usize..MAX_INIT_PROG_SIZE),
                    &reg_config
                )
            )
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
        let max_generations = 10000*self.programs.len();

        for _ in 0..max_generations {
            println!("{}", self);
            let results: TournamentResult = self.tournament_selection();
            self.update_population(&results);
        }

        // Return the best performing program
        self.programs[self.training_best_index].clone()
    }

    // Select random programs, split group in half, and select
    // winner from each group and loser from each group.
    fn tournament_selection(&mut self) -> TournamentResult {
        // Select the random individuals. This contains the original 
        // index for each chosen individual
        let competitor_indices: Vec<usize> = select_no_replacement(
            self.programs.len(),
            2 * TOURNAMENT_SIZE
        );

        // Store the fitness values for each of the chosen. Note, 
        // the 0th element in this array corresponds to the 0th
        // element in `competitor_indices` which contains the 
        // original index (wrt self.programs) of the first selected
        // program.
        let competitors: Vec<f64> = competitor_indices
            .iter()
            .map(|x| self.fitness_values[*x])
            .collect();

        // Split the group into two halves
        let first_group: &[f64] = &competitors[..TOURNAMENT_SIZE];
        let second_group: &[f64] = &competitors[TOURNAMENT_SIZE..(2*TOURNAMENT_SIZE)];

        // Get the winners and losers
        let first_winner_index: usize = smallest_element_index(first_group);
        let first_loser_index: usize = largest_element_index(first_group);
        let second_winner_index: usize = smallest_element_index(first_group);
        let second_loser_index: usize = largest_element_index(second_group);

        // Get back the indices (wrt self.programs) corresponding to the 
        // winners and losers
        let first_winner = competitor_indices[first_winner_index];
        let first_loser = competitor_indices[first_loser_index];
        let second_winner = competitor_indices[second_winner_index + TOURNAMENT_SIZE];
        let second_loser = competitor_indices[second_loser_index + TOURNAMENT_SIZE];

        TournamentResult {
            winners: [first_winner, second_winner],
            losers: [first_loser, second_loser]
        }
    }
    
    // Takes the two winners of the last tournament, copies them, then
    // performs either crossover or mutation. After undergoing one
    // of the variation operators, the modified winners will replace
    // the spots in the population where the original winners were.
    //
    // For the two losers of the tournament, either they or the original 
    // winners will be put back into the population. This is chosen based
    // on a probability.
    fn update_population(&mut self, results: &TournamentResult) {
        // Store the indicies of the winners and losers
        let winner_index1: usize = results.winners[0];
        let winner_index2: usize = results.winners[1];
        let loser_index1: usize = results.losers[0];
        let loser_index2: usize = results.losers[1];

        // Clone the original winners and their fitness values
        let original_winner1: Program = self.programs[winner_index1].clone();
        let original_winner2: Program = self.programs[winner_index2].clone();
        let original_winner1_fitness: f64 = self.fitness_values[winner_index1];
        let original_winner2_fitness: f64 = self.fitness_values[winner_index2];

        // Create two new Programs by applying a variation
        // operator to the last tournament's winners
        let mut rng = rand::rng();
        let new_members: [Program; 2] = if rng.random::<f64>() < CROSSOVER_RATE {
            self.crossover(winner_index1, winner_index2)
        } else {
            [self.mutate(winner_index1), self.mutate(winner_index2)]
        };

        // Replace the original winners with the new members
        self.programs[winner_index1] = new_members[0].clone();
        self.programs[winner_index2] = new_members[1].clone();

        // Update the corresponding fitness value entries
        self.fitness_values[winner_index1] = mse(
            &mut self.programs[winner_index1],
            &self.training_data
        );
        self.fitness_values[winner_index2] = mse(
            &mut self.programs[winner_index2],
            &self.training_data
        );

        // Check if either new member usurps the current training best
        if self.fitness_values[winner_index1] < self.fitness_values[self.training_best_index] {
            self.training_best_index = winner_index1;
        }
        if self.fitness_values[winner_index2] < self.fitness_values[self.training_best_index] {
            self.training_best_index = winner_index2;
        }

        // TODO: Add validation stuff here
        
        // Perform reproduction of original winners or let
        // the last tournament's losers stay in the population
        if rng.random::<f64>() < REPRODUCTION_RATE {
            self.programs[loser_index1] = original_winner1;
            self.programs[loser_index2] = original_winner2;

            // Update the corresponding fitness values
            self.fitness_values[loser_index1] = original_winner1_fitness;
            self.fitness_values[loser_index2] = original_winner2_fitness;
        }
    }

    fn crossover(
        &self, 
        parent1_index: usize, 
        parent2_index: usize
    ) -> [Program; 2] {
        todo!()
    }

    fn mutate(&self, index: usize) -> Program {
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

        //writeln!(f, "--------------------------------------")?;
        writeln!(f, "Average Fitness Value: {avg_fitness_value:.2}")?;
        writeln!(f, "Average Program Length: {avg_prog_len:.1}")?;
        writeln!(f, "Best Training Fitness: {:.3}", self.fitness_values[self.training_best_index])?;
        writeln!(f, "Best Training Length: {}", self.programs[self.training_best_index].instructions.len())
        //writeln!(f, "Best Validation Fitness: {}", self.fitness_values[self.validation_best_index])?;
        //write!(f, "Best Validation Length: {}", self.programs[self.validation_best_index].instructions.len())?;
        //write!(f, "\n--------------------------------------")
    }
}
