use lgp::program::{ProgramConfig};
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

    let prog_config = ProgramConfig {
        total_var_registers: 8,
        total_const_registers: 50,
        const_start: -10.0,
        const_step_size: 0.5,
        input_register: 1,
        output_register: 0,
        initial_var_value: 0.0,
        max_seg_len: 5,
        max_cp_dist: 20,
        max_seg_diff: 1,
        mutation_step_size: 1,
        min_prog_len: 2,
        max_prog_len: 200
    };

    let pop_config = PopulationConfig {
        population_size: 1000,
        max_init_prog_size: 20,
        crossover_rate: 0.78,
        sa_step_size: 0.1,
        learning_rate: 0.10,
        reproduction_rate: 0.69,
        tournament_size: 4,
        prog_config
    };
   

    let mut pop = Population::new(training_data, validation_data, pop_config);
}
