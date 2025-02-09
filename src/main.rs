use lgp::population::Population;
//use lgp::program::Program;

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
    
    let pop_size: usize = 20;
    let pop = Population::new(pop_size, training_data, validation_data);

    println!("{}", pop);
}
