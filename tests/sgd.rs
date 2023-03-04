extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::optim::{Optimizer, SGD};
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [1., 1., 1., 1., 1.];
const GRADS: [f64; 5] = [1., 1., 1., 1., 1.];

fn build_params() -> Vec<Value> {
    VALUES
        .into_iter()
        .zip(GRADS)
        .map(|(val, grad)| {
            let value = Value::from(val);
            *value.grad_mut() = Value::from(grad);
            value
        })
        .collect()
}

fn assert_params(params: &Vec<Value>, actuals: Vec<f64>) {
    for (param, actual) in params.iter().zip(actuals) {
        assert_abs_diff_eq!(param.value(), actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_sgd_learning_rate_update() {
    let params = build_params();

    let mut optim = SGD {
        params: params.clone(),
        lr: 2.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![-1.0; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![-3.0; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_sgd_learning_rate_maximize_update() {
    let params = build_params();

    let mut optim = SGD {
        params: params.clone(),
        lr: 2.0,
        maximize: true,
        ..Default::default()
    };

    optim.step();
    let actuals = vec![3.0; 5];
    assert_params(&params, actuals);
}

#[test]
fn valid_sgd_learning_rate_with_momentum_update() {
    let params = build_params();

    let mut optim = SGD {
        params: params.clone(),
        lr: 2.0,
        momentum: 1.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![-1.0; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![-5.0; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_sgd_learning_rate_with_dampened_momentum_update() {
    let params = build_params();

    let mut optim = SGD {
        params: params.clone(),
        lr: 1.0,
        dampening: 0.5,
        momentum: 1.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.0; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![-1.5; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_sgd_learning_rate_with_weight_decay_update() {
    let params = build_params();

    let mut optim = SGD {
        params: params.clone(),
        lr: 2.0,
        weight_decay: 2.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![-5.0; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![13.0; 5];
    assert_params(&params, second_step);
}
