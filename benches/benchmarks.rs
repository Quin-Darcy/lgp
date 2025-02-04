#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use lgp::program::Program;

fn benchmark_small_program_creation(c: &mut Criterion) {
    c.bench_function("create_small_program", |b| {
        b.iter(|| {
            Program::new(10)
        })
    });
}


fn benchmark_medium_program_creation(c: &mut Criterion) {
    c.bench_function("create_medium_program", |b| {
        b.iter(|| {
            Program::new(1000)
        })
    });
}


fn benchmark_large_program_creation(c: &mut Criterion) {
    c.bench_function("create_large_program", |b| {
        b.iter(|| {
            Program::new(10000)
        })
    });
}

fn benchmark_small_program_run(c: &mut Criterion) {
    let mut p = Program::new(10);
    c.bench_function("run_small_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}


fn benchmark_medium_program_run(c: &mut Criterion) {
    let mut p = Program::new(1000);
    c.bench_function("run_medium_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}


fn benchmark_large_program_run(c: &mut Criterion) {
    let mut p = Program::new(10000);
    c.bench_function("run_large_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}

criterion_group!(benches,
    benchmark_small_program_creation,
    benchmark_medium_program_creation,
    benchmark_large_program_creation,
    benchmark_small_program_run,
    benchmark_medium_program_run,
    benchmark_large_program_run
);
criterion_main!(benches);
