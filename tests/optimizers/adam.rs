extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::optim::{Adam, Optimizer};
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [1., 1., 1., 1., 1.];
const STARTING_GRADS: [f64; 5] = [1., 1., 1., 1., 1.];

fn build_params() -> Vec<Value> {
    VALUES
        .into_iter()
        .zip(STARTING_GRADS)
        .map(|(val, grad)| {
            let value = Value::from(val);
            *value.grad_mut() = Value::from(grad);
            value
        })
        .collect()
}

fn set_grads(values: &Vec<Value>, grad: f64) {
    for value in values.iter() {
        *value.grad_mut() = val!(grad);
    }
}

fn assert_params(params: &Vec<Value>, actuals: Vec<f64>) {
    for (param, actual) in params.iter().zip(actuals) {
        assert_abs_diff_eq!(param.value(), actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_adam_learning_rate_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![0.9683052; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_without_first_moment_decay_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        betas: (0.0, 0.999),
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![1.013236; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_without_second_moment_decay_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        betas: (0.9, 0.0),
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![0.9644737; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_without_moment_decay_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        betas: (0.0, 0.0),
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![1.0; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_with_eps_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        eps: 1e-4,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.950005; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![0.968309; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_with_weight_decay_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        weight_decay: 2.0,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![0.9177558; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_with_amsgrad_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        amsgrad: true,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![0.95; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![0.9683052; 5];
    assert_params(&params, second_step);
}

#[test]
fn valid_adam_learning_rate_with_maximize_update() {
    let params = build_params();

    let mut optim = Adam {
        params: params.clone(),
        lr: val!(0.05),
        maximize: true,
        ..Default::default()
    };

    optim.step();
    let first_step = vec![1.05; 5];
    assert_params(&params, first_step);

    set_grads(&params, -2.0);

    optim.step();
    let second_step = vec![1.0316948; 5];
    assert_params(&params, second_step);
}
