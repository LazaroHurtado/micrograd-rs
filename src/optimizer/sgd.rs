use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct SGDCache {
    pub time_step: usize,
    pub prev_gradients: Option<Array1<f64>>,
}

pub struct SGDConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub maximize: bool,
    pub cache: SGDCache,
}

impl Default for SGDConfig {
    fn default() -> Self {
        SGDConfig {
            lr: 0.01,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            maximize: false,
            cache: Default::default(),
        }
    }
}

impl Optimizer {
    pub fn sgd(&mut self) {
        if let Self::SGD(params, config) = self {
            let params_n = params.len();
            let SGDCache {
                ref mut time_step,
                ref mut prev_gradients,
            } = config.cache;

            let prev_grads = prev_gradients.get_or_insert(Array1::from_vec(vec![0.; params_n]));

            for (i, param) in params.iter_mut().enumerate() {
                let mut grad = param.grad().unwrap_or_else(Value::zero).value();
                grad += param.value() * config.weight_decay;

                if *time_step > 0 {
                    prev_grads[i] =
                        (config.momentum * prev_grads[i]) + ((1. - config.dampening) * grad);
                } else {
                    prev_grads[i] = grad;
                }

                let step = config.lr * prev_grads[i];

                match config.maximize {
                    true => *param.value_mut() += step,
                    false => *param.value_mut() -= step,
                }
            }
            *time_step += 1;
        };
    }
}
