use super::{RMSPropConfig, SGDConfig};
use crate::prelude::*;

pub enum Optimizer {
    SGD(Tensor<Ix1>, SGDConfig),
    RMSProp(Tensor<Ix1>, RMSPropConfig),
}

impl Optimizer {
    pub fn step(&mut self) {
        match self {
            Self::SGD(_, _) => self.sgd(),
            Self::RMSProp(_, _) => self.rmsprop(),
        }
    }

    pub fn zero_grad(&mut self) {
        let params = match self {
            Self::SGD(params, _) => params,
            Self::RMSProp(params, _) => params,
        };

        for param in params {
            param.zero_grad();
        }
    }
}
