#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::complexity)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

//! A linear genetic programming implementation in Rust.
//! 
//! This library provides functionality for evolving programs that map inputs to outputs
//! using principles from linear genetic programming. Programs are represented as sequences
//! of register-based instructions that perform basic arithmetic operations.
//! 
//! # Architecture
//! 
//! The library is organized into two main modules:
//! 
//! - [`program`]: Defines the structure and execution of individual programs
//!
//! # Example
//!
//! ```
//! use lgp::Program;
//! 
//! // Create a program with 5 instructions
//! let mut program = Program::new(5);
//! 
//! // Run the program with input 2.0
//! let output = program.run(2.0);
//! # assert!(output.is_finite());
//! ```

pub mod program;
