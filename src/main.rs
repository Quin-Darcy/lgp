use lgp::population::{Population, PopulationConfig};


fn main() {
    /*
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
    */

    let training_data: Vec<(f64, f64)> = vec![
        (-10.0, -351.0),
        (-9.0, -289.0),
        (-8.0, -233.0),
        (-7.0, -183.0),
        (-6.0, -139.0),
        (-5.0, -101.0),
        (-4.0, -69.0),
        (-3.0, -43.0),
        (-2.0, -23.0),
        (-1.0, -9.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (2.0, -3.0),
        (3.0, -13.0),
        (4.0, -29.0),
        (5.0, -51.0),
        (6.0, -79.0),
        (7.0, -113.0),
        (8.0, -153.0),
        (9.0, -199.0),
        (10.0, -251.0)
    ];

    let validation_data: Vec<(f64, f64)> = vec![
        (11.0, -309.0),
        (12.0, -373.0),
        (13.0, -443.0),
        (14.0, -519.0),
        (15.0, -601.0),
        (16.0, -689.0),
        (17.0, -783.0)
    ];

    let mut pop_config = PopulationConfig::default();
    pop_config.prog_config.total_const_registers = 6;
    pop_config.prog_config.const_start = -3.0;
    pop_config.prog_config.const_step_size = 1.0;
    pop_config.prog_config.min_prog_len = 1;
    let mut pop = Population::new(training_data, validation_data, pop_config);
    pop.evolve();
}
