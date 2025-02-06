//! This module provides the main data structure and logic behind
//! the linear genetic program evolution. The main data structure 
//! `Population`, represents a group of `Programs` undergoing 
//! evolution with respect to some fitness function.
use crate::program::Program;


/// Main structure for the management and evolution of the programs.
///
/// The members of a `Population` instance are:
/// - `programs`: Vector containing the `Program`s undergoing evolution.
/// - `fitness_values`: The fitness value for each of the programs.
/// - `training_best`: Index to the program that has performed best on the given training data.
/// - `validation_best`: Index to the program that has performed best on unknown data.
pub struct Population {
    programs: Vec<Program>,
    fitness_values: Vec<f64>,
    training_best: usize,
    validation_best: usize
}
