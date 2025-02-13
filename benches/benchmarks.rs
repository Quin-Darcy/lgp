#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};
use lgp::program::{Program, RegisterConfig};

const CONFIG: RegisterConfig = RegisterConfig {
    total_var_registers: 8,
    total_const_registers: 100,
    const_start: -50.0,
    const_step_size: 1.0,
    input_register: 1,
    output_register: 0,
    initial_var_value: 1.0
};

fn benchmark_small_program_creation(c: &mut Criterion) {
    c.bench_function("create_small_program", |b| {
        b.iter(|| {
            Program::new(10, &CONFIG)
        })
    });
}


fn benchmark_medium_program_creation(c: &mut Criterion) {
    c.bench_function("create_medium_program", |b| {
        b.iter(|| {
            Program::new(1000, &CONFIG)
        })
    });
}


fn benchmark_large_program_creation(c: &mut Criterion) {
    c.bench_function("create_large_program", |b| {
        b.iter(|| {
            Program::new(10000, &CONFIG)
        })
    });
}

fn benchmark_small_program_run(c: &mut Criterion) {
    let mut p = Program::new(10, &CONFIG);
    c.bench_function("run_small_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}


fn benchmark_medium_program_run(c: &mut Criterion) {
    let mut p = Program::new(1000, &CONFIG);
    c.bench_function("run_medium_program", |b| {
        b.iter(|| {
            p.run(2.0)
        })
    });
}


fn benchmark_large_program_run(c: &mut Criterion) {
    let mut p = Program::new(10000, &CONFIG);
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
