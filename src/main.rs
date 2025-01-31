use lgp::program::Program;

fn main() {
    let initial_size: usize = 8;
    let mut program = Program::new(initial_size);
    println!("{:?}", program.run(0.0));
}
