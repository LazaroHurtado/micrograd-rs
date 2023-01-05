extern crate micrograd_rs;
use micrograd_rs::optim::{Optimizer, SGDCache, SGDConfig};
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [1., 1., 1., 1., 1.];
const GRADS: [f64; 5] = [1., 1., 1., 1., 1.];
const PREV_GRADS: [f64; 5] = [2., 2., 2., 2., 2.];

fn build_params() -> Tensor<Ix1> {
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

#[test]
fn valid_sgd_learning_rate_minimize_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 2.0,
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), -1.0);
    }
}

#[test]
fn valid_sgd_learning_rate_maximize_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 2.0,
            maximize: true,
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), 3.0);
    }
}

#[test]
fn valid_sgd_learning_rate_maximize_with_empty_momentum_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 2.0,
            momentum: 0.1,
            maximize: true,
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), 3.0);
    }
}

#[test]
fn valid_sgd_learning_rate_maximize_with_momentum_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 2.0,
            momentum: 1.0,
            cache: SGDCache {
                prev_gradients: Some(Array1::from_vec(PREV_GRADS.to_vec())),
            },
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), -5.0);
    }
}

#[test]
fn valid_sgd_learning_rate_maximize_with_dampend_momentum_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 1.0,
            dampening: 0.5,
            momentum: 1.0,
            cache: SGDCache {
                prev_gradients: Some(Array1::from_vec(PREV_GRADS.to_vec())),
            },
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), -1.5);
    }
}

#[test]
fn valid_sgd_learning_rate_maximize_with_weight_decay_update() {
    let params = build_params();

    let mut optim = Optimizer::SGD(
        params.clone(),
        SGDConfig {
            lr: 2.0,
            weight_decay: 2.0,
            ..Default::default()
        },
    );
    optim.step();

    for param in params {
        assert_eq!(param.value(), -5.0);
    }
}
