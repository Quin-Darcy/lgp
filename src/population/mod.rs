//! Structure containing the population and methods for evolving the population.
//!
//! This module provides the main data structure whose methods implement the genetic algorithm
//! which evolves the programs. An instance of a Population consists of the following fields:
//!
//! - `individuals`: This is a vector of Programs. 
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

use crate::program::Program;

/// This is the initial size used for each Program in the set of individuals.
const PROGRAM_SIZE: usize = 20; 

/// Structure containing the population of Programs.
pub struct Population {
    individuals: Vec<Program>
}

impl Population {
    /// Create new population of Programs.
    ///
    /// # Arguments
    /// * `population_size`: The number of individuals in the population.
    pub fn new(population_size: usize) -> Self {
        let mut indvs: Vec<Program> = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            indvs.push(Program::new(PROGRAM_SIZE));
        }

        Self {
            individuals: indvs
        }
    }
}
