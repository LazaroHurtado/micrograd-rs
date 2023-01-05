use super::SGDConfig;
use crate::prelude::*;

pub enum Optimizer {
    SGD(Tensor<Ix1>, SGDConfig),
}

impl Optimizer {
    pub fn step(&mut self) {
        match self {
            Self::SGD(_, _) => self.sgd(),
        }
    }

    pub fn zero_grad(&mut self) {
        let params = match self {
            Self::SGD(params, _) => params,
        };

        for param in params {
            param.zero_grad();
        }
    }
}
