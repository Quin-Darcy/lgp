#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

//! # Overview
//!
//! A linear genetic programming implementation in Rust.
//! 
//! This library provides functionality for evolving programs that map inputs to outputs
//! using principles from linear genetic programming. The purpose of linear genetic programming is
//! to evolve a program which represents the solution to some sort of problem. Programs are represented as sequences
//! of register-based instructions that perform basic arithmetic operations. 
//!
//! **Example.** Suppose you have a finite set of (x, y) pairs. These data points may have come
//! from some unknown polynomial. We can attempt to evolve a program which simulates the behavior
//! of this polynomial by selecting those programs which best fit the given data. The resultant
//! program would therefore be an approximate, but functional representation of the unknown polynomial.
//!
//! ### General Evolutionary Algorithm
//! 1. Randomly initialize a popuation of individual programs.
//! 2. Select individuals from the population that are fitter than others by using a certain
//!    *selection method*. The *fitness measure* defines the problem the algorithm is expected to
//!    solve.
//! 3. Generate new variants by applying the following *genetic operators* with certain
//!    probabilities:
//!    - *Reproduction*: Copy an individual without change.
//!    - *Recombination*: Randomly exchange substructers between individuals.
//!    - *Mutation*: Randomly replace a substructure in an individual.
//! 4. If the termination criterion is not met, go back to step 2.
//! 5. Stop. The best individual represents the best solution found.
//! 
//! # Architecture
//! 
//! The library is organized into two main modules:
//! 
//! - [`program`]: Defines the structure and execution of individual programs
//! - [`population`]: Structures and methods behind main evolution engine.
//!
//! # Example
//!
//! ```
//! use lgp::Population;
//! use lgp::Program;
//! 
//! // Example training data
//! let training_data: Vec<(f64, f64)> = vec![(1.0, 2.0), (2.0, 4.0)];
//!
//! // Create a population of 40 Programs
//! let mut pop = Population::new(40);
//! 
//! // Evolve a program against training data
//! let mut prog: Program = pop.evolve(&training_data)
//! ```

pub mod program;
//pub mod population;
