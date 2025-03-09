use lgp::population::{Population, PopulationConfig};


fn main() {
    let training_data: Vec<(f64, f64)> = vec![
        (-5.0, 25.0),
        (-4.0, 16.0),
        (-3.0, 9.0),
        (-2.0, 4.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0), 
        (2.0, 4.0),
        (3.0, 9.0),
        (4.0, 16.0),
        (5.0, 25.0)
    ];

    let validation_data: Vec<(f64, f64)> = vec![
        (-10.0, 100.0),
        (-9.0, 81.0),
        (-8.0, 64.0),
        (-7.0, 49.0),
        (-6.0, 36.0)
    ];

    let mut pop_config = PopulationConfig::default();
    pop_config.prog_config.total_const_registers = 10;
    pop_config.prog_config.const_start = -1.0;
    pop_config.prog_config.const_step_size = 0.2;
    pop_config.prog_config.min_prog_len = 1;
    let mut pop = Population::new(training_data, validation_data, pop_config);
    pop.evolve();
}
