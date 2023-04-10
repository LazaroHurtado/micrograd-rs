use super::Optimizer;
use crate::prelude::*;

#[derive(Default)]
pub struct AdamCache {
    pub time_step: usize,
    pub exp_avgs: Option<Array1<f64>>,
    pub exp_avg_sqs: Option<Array1<f64>>,
    pub max_exp_avg_sqs: Option<Array1<f64>>,
}

pub struct Adam {
    pub params: Vec<Value>,
    pub lr: Value,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub maximize: bool,
    pub cache: AdamCache,
}

impl Default for Adam {
    fn default() -> Self {
        Adam {
            params: vec![],
            lr: val!(0.001),
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            cache: Default::default(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        let params_n = self.params.len();
        let (beta1, beta2) = self.betas;
        let AdamCache {
            ref mut time_step,
            ref mut exp_avgs,
            ref mut exp_avg_sqs,
            ref mut max_exp_avg_sqs,
        } = self.cache;

        let exp_avgs = exp_avgs.get_or_insert(Array1::from_vec(vec![0.; params_n]));
        let exp_avg_sqs = exp_avg_sqs.get_or_insert(Array1::from_vec(vec![0.; params_n]));
        let max_exp_avg_sqs = max_exp_avg_sqs.get_or_insert(Array1::from_vec(vec![0.; params_n]));

        for (i, param) in self.params.iter().enumerate() {
            let mut grad = param
                .grad()
                .unwrap_or_else(|| panic!("Optimizer cannot step when gradient is None."))
                .value();

            if self.maximize {
                grad = -grad;
            }
            grad += param.value() * self.weight_decay;

            exp_avgs[i] = (beta1 * exp_avgs[i]) + ((1. - beta1) * grad);
            exp_avg_sqs[i] = (beta2 * exp_avg_sqs[i]) + ((1. - beta2) * grad.powf(2.0));

            let t = 1 + (*time_step as i32);
            let bias_correction1 = 1.0 - beta1.powi(t);
            let bias_correction2 = 1.0 - beta2.powi(t);

            let step_size = self.lr.value() / bias_correction1;

            let bias_correction2_sqrt = bias_correction2.sqrt();

            let denom = if self.amsgrad {
                max_exp_avg_sqs[i] = max_exp_avg_sqs[i].max(exp_avg_sqs[i]);
                (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt) + self.eps
            } else {
                (exp_avg_sqs[i].sqrt() / bias_correction2_sqrt) + self.eps
            };
            let step = (exp_avgs[i] / denom) * step_size;

            *param.value_mut() -= step;
        }
        *time_step += 1;
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
