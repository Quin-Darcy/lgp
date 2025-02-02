#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use lgp::program::Program;
use lgp::population::Population;

fn benchmark_population_creation(c: &mut Criterion) {
    c.bench_function("create_population", |b| {
        b.iter(|| {
            Population::new(100)
        })
    });
}

fn benchmark_population_eval_fitness(c: &mut Criterion) {
    let mut pop = Population::new(100);
    let training_data: Vec<(f64, f64)> = vec![(2.3, 3.4), (18.9, 23.6), (-10.2, -0.01), (4.7, -0.03)];
    c.bench_function("eval_fitness", |b| {
        b.iter(|| {
            pop.eval_fitness(&training_data)
        })
    });
}

fn benchmark_program_creation(c: &mut Criterion) {
    c.bench_function("create_program", |b| {
        b.iter(|| {
            Program::new(100)
        })
    });
}

fn benchmark_program_run(c: &mut Criterion) {
    let mut p = Program::new(100);
    c.bench_function("run_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}

criterion_group!(benches,
    //benchmark_population_creation,
    benchmark_population_eval_fitness
    //benchmark_program_creation,
    //benchmark_program_run
);
criterion_main!(benches);
