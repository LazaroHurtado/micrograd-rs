use num_traits::Pow;

use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct RMSPropCache {
    pub prev_gradients: Option<Array1<f64>>,
    pub moving_avg: Option<Array1<f64>>,
    pub avg_gradients: Option<Array1<f64>>,
}

pub struct RMSPropConfig {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub centered: bool,
    pub maximize: bool,
    pub cache: RMSPropCache,
}

impl Default for RMSPropConfig {
    fn default() -> Self {
        RMSPropConfig {
            lr: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            centered: false,
            maximize: false,
            cache: Default::default(),
        }
    }
}

impl Optimizer {
    pub fn rmsprop(&mut self) {
        if let Self::RMSProp(params, config) = self {
            let params_n = params.len();
            let cache = &mut config.cache;

            let prev_grads = cache
                .prev_gradients
                .get_or_insert(Array1::from_vec(vec![0.; params_n]));
            let moving_avg = cache
                .moving_avg
                .get_or_insert(Array1::from_vec(vec![0.; params_n]));
            let avg_gradients = cache
                .avg_gradients
                .get_or_insert(Array1::from_vec(vec![0.; params_n]));

            for (t, param) in params.iter_mut().enumerate() {
                let mut grad = param.grad().unwrap_or_else(Value::zero).value();
                grad += param.value() * config.weight_decay;

                moving_avg[t] =
                    (config.alpha * moving_avg[t]) + ((1. - config.alpha) * grad.pow(2));
                let mut curr_moving_avg = moving_avg[t];

                if config.centered {
                    avg_gradients[t] =
                        (config.alpha * avg_gradients[t]) + ((1. - config.alpha) * grad);
                    curr_moving_avg -= avg_gradients[t].pow(2);
                }

                let momentum = config.momentum * prev_grads[t];
                prev_grads[t] = momentum + (grad / (curr_moving_avg.sqrt() + config.eps));
                let step = config.lr * prev_grads[t];

                match config.maximize {
                    true => *param.value_mut() += step,
                    false => *param.value_mut() -= step,
                }
            }
        }
    }
}
