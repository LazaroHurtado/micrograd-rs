use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct SGDCache {
    pub prev_gradients: Option<Array1<f64>>,
}

#[derive(Default)]
pub struct SGDConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub maximize: bool,
    pub cache: SGDCache,
}

impl Optimizer {
    pub fn sgd(&mut self) {
        let Self::SGD(params, config) = self;
        let mut grads = params.mapv(|param| param.grad_mut().clone());

        grads += &(&*params * config.weight_decay);

        grads = match &config.cache.prev_gradients {
            None => {
                config.cache.prev_gradients = Some(grads.mapv(|grad| grad.value()));

                grads
            }
            Some(prev_grads) => {
                let grads_with_momentum =
                    (&grads * (1.0 - config.dampening)) + (&*prev_grads * config.momentum);
                config.cache.prev_gradients =
                    Some(grads_with_momentum.mapv(|grad_with_momentum| grad_with_momentum.value()));

                grads_with_momentum
            }
        };

        grads = grads * config.lr;

        for (param, grad) in params.iter().zip(grads) {
            match config.maximize {
                true => *param.value_mut() += grad.value(),
                false => *param.value_mut() -= grad.value(),
            }
        }
    }
}
