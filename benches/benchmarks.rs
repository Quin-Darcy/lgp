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
    benchmark_population_creation,
    //benchmark_program_creation,
    //benchmark_program_run
);
criterion_main!(benches);
