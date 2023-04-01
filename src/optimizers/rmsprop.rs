use num_traits::Pow;

use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct RMSPropCache {
    pub prev_gradients: Option<Array1<f64>>,
    pub moving_avg: Option<Array1<f64>>,
    pub avg_gradients: Option<Array1<f64>>,
}

pub struct RMSProp {
    pub params: Vec<Value>,
    pub lr: Value,
    pub alpha: f64,
    pub eps: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub centered: bool,
    pub maximize: bool,
    pub cache: RMSPropCache,
}

impl Default for RMSProp {
    fn default() -> Self {
        RMSProp {
            params: vec![],
            lr: val!(0.01),
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

impl Optimizer for RMSProp {
    fn step(&mut self) {
        let params_n = self.params.len();
        let RMSPropCache {
            ref mut prev_gradients,
            ref mut moving_avg,
            ref mut avg_gradients,
        } = self.cache;

        let prev_grads = prev_gradients.get_or_insert(Array1::from_vec(vec![0.; params_n]));
        let moving_avg = moving_avg.get_or_insert(Array1::from_vec(vec![0.; params_n]));
        let avg_gradients = avg_gradients.get_or_insert(Array1::from_vec(vec![0.; params_n]));

        for (i, param) in self.params.iter().enumerate() {
            let mut grad = param
                .grad()
                .unwrap_or_else(|| panic!("Optimizer cannot step when gradient is None."))
                .value();
            grad += param.value() * self.weight_decay;

            moving_avg[i] = (self.alpha * moving_avg[i]) + ((1. - self.alpha) * grad.pow(2));
            let mut curr_moving_avg = moving_avg[i];

            if self.centered {
                avg_gradients[i] = (self.alpha * avg_gradients[i]) + ((1. - self.alpha) * grad);
                curr_moving_avg -= avg_gradients[i].pow(2);
            }

            let momentum = self.momentum * prev_grads[i];
            prev_grads[i] = momentum + (grad / (curr_moving_avg.sqrt() + self.eps));
            let step = self.lr.value() * prev_grads[i];

            match self.maximize {
                true => *param.value_mut() += step,
                false => *param.value_mut() -= step,
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter() {
            param.zero_grad();
        }
    }

    fn lr(&self) -> Value {
        self.lr.clone()
    }
}
