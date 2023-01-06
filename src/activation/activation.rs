#![allow(clippy::module_inception)]

use crate::modules::Module;
use crate::prelude::*;

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    ReLu,
    Tanh,
    Softmax,
}

impl Activation {
    pub fn activate<D: Dimension>(&self, inputs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::ReLu => inputs.mapv(|input| self.relu(input)),
            Self::Tanh => inputs.mapv(|input| self.tanh(input)),
            Self::Softmax => self.softmax(inputs),
        }
    }
}

impl<D> Module<D> for Activation
where
    D: Dimension,
{
    fn forward(&self, outputs: Tensor<D>) -> Tensor<D> {
        self.activate(outputs)
    }
}
