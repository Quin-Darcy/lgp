//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The `Population` data structure 
//! represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::{Program, RegisterConfig};
use rand::Rng;
use std::{cmp, fmt};

mod utilities;
use utilities::{
    mse, 
    smallest_element_index,
    largest_element_index,
    select_no_replacement
}; 

/// Struct defining parameters controlling population and evolution
#[allow(clippy::module_name_repetitions)]
pub struct PopulationConfig {
    /// Number of programs in population
    pub population_size: usize,
    /// Max number of instructions new programs can be initialized with
    pub max_init_prog_size: usize,
    /// Sets probability of variation operator being crossover
    pub crossover_rate: f64,
    /// Maximum segment length
    pub max_seg_len: usize,
    /// Maximum distance between crossover points
    pub max_cp_dist: usize,
    /// Maximum difference in segment lengths
    pub max_seg_diff: usize,
    /// Number of instructions that can be mutated in a single variation
    pub mutation_step_size: usize,
    /// Sets the step size for self-adapdation
    pub sa_step_size: f64,
    /// Sets the rate which the coevolving variation parameters mutate
    pub learning_rate: f64,
    /// Rate at which winners overwrite losers of tournaments
    pub reproduction_rate: f64,
    /// Number of programs to participate in tournament
    pub tournament_size: usize,
    /// Minimum program length
    pub min_prog_len: usize,
    /// Maximum program length
    pub max_prog_len: usize,
    /// Register configuration struct
    pub reg_config: RegisterConfig
}

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
    mutation_parameters: Vec<f64>,
    training_best_index: usize,
    validation_best_index: usize,
    training_data: Vec<(f64, f64)>,
    validation_data: Vec<(f64, f64)>,
    config: PopulationConfig
}

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
        training_data: Vec<(f64, f64)>,
        validation_data: Vec<(f64, f64)>,
        config: PopulationConfig
    ) -> Self {
        // Initialize population as set of random Programs
        let mut rng = rand::rng();
        let pop_size = if config.population_size % 2 == 0 { 
            config.population_size 
        } else {
            config.population_size + 1
        };
        let mut programs: Vec<Program> = (0..pop_size)
            .map(|_| Program::new(
                    rng.random_range(2_usize..config.max_init_prog_size),
                    &config.reg_config
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
            mutation_parameters: vec![0.5; pop_size],
            training_best_index,
            validation_best_index: 0,
            training_data,
            validation_data,
            config
        }
    }

    /// Main genetic algorithm loop to evolve a program against the training data.
    pub fn evolve(&mut self) -> Program {
        /*
         * The main loop consists of the following steps:
         * 1. Peform selectoin to return two winners and two losers.
         * 2. Copy winners.
         * 3. Mutate or crossover winners and add to population.
         *    - Probabilistically mutate the mutation parameters
         * 4. Probabilistically replace losers with original winners.
         * 5. Update training and validation bests.
         */

        // Two old programs will be replaced every generation with
        // two new programs. Therefore after the number of generations
        // elapsed is equal to have the population size, all original
        // programs in the population will have been replaced.
        let max_generations = 2*self.programs.len();

        for _ in 0..max_generations {
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
            2 * self.config.tournament_size
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
        let first_group: &[f64] = &competitors[..self.config.tournament_size];
        let second_group: &[f64] = &competitors[self.config.tournament_size..(2*self.config.tournament_size)];

        // Get the winners and losers
        let first_winner_index: usize = smallest_element_index(first_group);
        let first_loser_index: usize = largest_element_index(first_group);
        let second_winner_index: usize = smallest_element_index(first_group);
        let second_loser_index: usize = largest_element_index(second_group);

        // Get back the indices (wrt self.programs) corresponding to the 
        // winners and losers
        let first_winner = competitor_indices[first_winner_index];
        let first_loser = competitor_indices[first_loser_index];
        let second_winner = competitor_indices[second_winner_index + self.config.tournament_size];
        let second_loser = competitor_indices[second_loser_index + self.config.tournament_size];

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

        // Update the self-adaptation mutation parameters of the winners
        self.update_parameters(winner_index1, winner_index2);

        // Create two new Programs by applying a variation
        // operator to the last tournament's winners
        let mut rng = rand::rng();
        let new_members: [Program; 2] = if rng.random::<f64>() < self.config.crossover_rate {
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
        if rng.random::<f64>() < self.config.reproduction_rate {
            self.programs[loser_index1] = original_winner1;
            self.programs[loser_index2] = original_winner2;

            // Update the corresponding fitness values
            self.fitness_values[loser_index1] = original_winner1_fitness;
            self.fitness_values[loser_index2] = original_winner2_fitness;
        }
    }

    // Perform probabilistic mutation of the winner's mutation parameters.
    //
    // The idea is that each program carries with it a parameter that affects
    // how it undergoes mutation. The parameter is the probability that
    // a micro-mutation or macro-mutation is selected. This probability is 
    // itself mutated at a rate set by the learning_rate parameter. This 
    // is an example of self-adapdation.
    fn update_parameters(&mut self, index1: usize, index2: usize) {
        let mut rng = rand::rng();
        
        // Early return if no update needed
        if rng.random::<f64>() >= self.config.learning_rate {
            return;
        }

        // Helper closure to update a single parameter
        let mut update_param = |param: &mut f64| {
            let step = if rng.random::<f64>() < 0.5 { 
                self.config.sa_step_size 
            } else { 
                -self.config.sa_step_size
            };
            let new_value = *param + step;
            if (0.0..=1.0).contains(&new_value) {
                *param = new_value;
            }
        };

        // Update both parameters
        update_param(&mut self.mutation_parameters[index1]);
        update_param(&mut self.mutation_parameters[index2]);
    }

    fn crossover(
        &self, 
        parent1_index: usize, 
        parent2_index: usize
    ) -> [Program; 2] {
        // Store the lengths of each parent program
        let prog1_len = self.programs[parent1_index].instructions.len();
        let prog2_len = self.programs[parent2_index].instructions.len();
    
        // Find the smaller of the two programs
        let mut prog_index1;
        let mut prog_index2;
        if prog1_len < prog2_len {
            prog_index1 = parent1_index;
            prog_index2 = parent2_index;
        } else {
            prog_index1 = parent2_index;
            prog_index2 = parent1_index;
        }

        let smaller_len = self.programs[prog_index1].instructions.len();
        let larger_len = self.programs[prog_index2].instructions.len();

        // Note that exchanging segments has the potential to alter a 
        // program's size by, at most, max_seg_diff. We need to assure 
        // that if the program's size is reduced, it doesn't fall below
        // min_prog_len. Similarly, if the program size is increased, it
        // must not exceed max_prog_len. 
        //
        // Both these cases can be avoided by accounting for 'how far'
        // the current program sizes are away from the min and max. 
        // With that, we can make sure to adjust max_seg_diff so that
        // swapping segments does not result in crossing either 
        // boundary.
        //
        // Ex. If prog1 has 10 instructions and the max_prog_len is 12
        // then we must make sure the max_seg_len <= 2. Similarly, if 
        // min_prog_len is 9, then max_seg_len <= 1. 
        //
        // We see if the program is closer in length to the max or the min.
        // We take the minimum between these two distances. We do this for
        // both programs. This gives us the largest delta either program
        // can change by while assuring neither will fall below or go over
        // either boundaries. We finally take the min between this delta 
        // and the max_seg_diff to give us the final allowed maximum
        // segment difference.
        let dist_from_min1: usize = smaller_len - self.config.min_prog_len;
        let dist_from_max1: usize = self.config.max_prog_len - smaller_len;
        let min1: usize = cmp::min(dist_from_min1, dist_from_max1);

        let dist_from_min2: usize = larger_len - self.config.min_prog_len;
        let dist_from_max2: usize = self.config.max_prog_len - larger_len;
        let min2: usize = cmp::min(dist_from_min2, dist_from_max2);
        
        // Changing the size of either program by this much or less is 
        // totally safe.
        let min_dist: usize = cmp::min(min1, min2);

        // Incorporate our parameter should it be smaller.
        let max_seg_diff: usize = cmp::min(min_dist, self.config.max_seg_diff);

        // Now we actually begin our choices
        let mut rng = rand::rng();

        // Select first crossover point from the smaller program
        let cp1: usize = rng.random_range(0..smaller_len - 1);

        // Select second crossover point from the second program
        // such that it remains in program bounds and the difference
        // between itself and cp1 does not exceed max_cp_dist
        let lower_cp: usize = cmp::max(0, cp1 - self.config.max_cp_dist);
        let upper_cp: usize = cmp::min(larger_len - 2, cp1 + self.config.max_cp_dist);
        let cp2: usize = rng.random_range(lower_cp..=upper_cp);

        // Calculate the remaining lengths between each crossover
        // point and the end of the program, for each program
        let remainder1: usize = smaller_len - cp1;
        let remainder2: usize = larger_len - cp2;

        // The minimum between both remainders gives an upper bound
        // which assures we don't generate segments which exceed either
        // program length
        let min_remainder: usize = cmp::min(remainder1, remainder2);

        // Select random first segment length
        let seg_len1: usize = rng.random_range(1..=min_remainder);

        // Select second segment length such that its difference 
        // is less than or equal to max_seg_diff
        let lower_seg: usize = cmp::max(1, seg_len1 - max_seg_diff);
        let upper_seg: usize = cmp::min(larger_len, seg_len1 + max_seg_diff);
        let seg_len2: usize = rng.random_range(lower_seg..=upper_seg);
        todo!()
    }

    /*
     *  Select up to mutation_step_size many instructions.
     *  For each of them, make a choice between a micro or macro
     *  mutation. Where micro refers to changing a constant or
     *  register. Macro refers to replacing the entire instruction
     *  with a random one, deleting an instruction, or adding an
     *  instruction.
     */
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
