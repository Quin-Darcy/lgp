use lgp::program::{Program, ProgramConfig};
use lgp::population::{Population, PopulationConfig};

fn print_program(p: &Program) {
    for inst in p.instructions.clone() {
        println!("0x{:x}", inst.0);
    }
}

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

    let prog_config = ProgramConfig::default();
    let prog1 = Program::new(20, &prog_config);
    let prog2 = Program::new(20, &prog_config);

    println!("Parent 1:");
    print_program(&prog1);

    println!("Parent 2:");
    print_program(&prog2);

    let offspring: [Program; 2] = prog1.crossover(&prog2.instructions);

    println!("Offspring 1:");
    print_program(&offspring[0]);

    println!("Offspring 2:");
    print_program(&offspring[1]);

    //let pop_config = PopulationConfig::default();
    //let pop = Population::new(training_data, validation_data, pop_config);
}
