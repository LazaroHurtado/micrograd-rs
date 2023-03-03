extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::optim::{Optimizer, RMSProp};
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
fn valid_rmsprop_learning_rate_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.01,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.9; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.82911193; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_rmsprop_learning_rate_maximize_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.01,
        maximize: true,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![1.1; 5];
    assert_params(&params, first_step);
}

#[test]
fn valid_rmsprop_learning_rate_with_momentum_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.01,
        momentum: 0.1,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.9; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.81911194; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_rmsprop_learning_rate_with_weight_decay_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 2.0,
        weight_decay: 2.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![-19.0; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.93523258; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_rmsprop_learning_rate_with_alpha_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.015,
        alpha: 0.85,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.96127015; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.9327954; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_rmsprop_learning_rate_with_eps_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.015,
        eps: 1e-4,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.85014987; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.743893; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_rmsprop_learning_rate_centered_update() {
    let params = build_params();

    let mut optim = RMSProp {
        params: params.clone(),
        lr: 0.015,
        weight_decay: 0.25,
        momentum: 0.1,
        alpha: 0.82,
        centered: true,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.9609566; 5];
    assert_params(&params, first_step);

    optim.step();
    let second_step = vec![0.9252056; 5];
    assert_params(&params, second_step);
}
