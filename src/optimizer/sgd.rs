use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct SGDCache {
    pub time_step: usize,
    pub prev_gradients: Option<Array1<f64>>,
}

pub struct SGD {
    pub params: Vec<Value>,
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub maximize: bool,
    pub cache: SGDCache,
}

impl Default for SGD {
    fn default() -> Self {
        SGD {
            params: vec![],
            lr: 0.01,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            maximize: false,
            cache: Default::default(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let params_n = self.params.len();
        let SGDCache {
            ref mut time_step,
            ref mut prev_gradients,
        } = self.cache;

        let prev_grads = prev_gradients.get_or_insert(Array1::from_vec(vec![0.; params_n]));

        for (i, param) in self.params.iter_mut().enumerate() {
            let mut grad = param
                .grad()
                .unwrap_or_else(|| panic!("Optimizer cannot step when gradient is None."))
                .value();
            grad += param.value() * self.weight_decay;

            if *time_step > 0 {
                prev_grads[i] = (self.momentum * prev_grads[i]) + ((1. - self.dampening) * grad);
            } else {
                prev_grads[i] = grad;
            }

            let step = self.lr * prev_grads[i];

            match self.maximize {
                true => *param.value_mut() += step,
                false => *param.value_mut() -= step,
            }
        }
        *time_step += 1;
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter() {
            param.zero_grad();
        }
    }
}
