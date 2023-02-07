#![allow(clippy::module_inception)]

use crate::layers::Layer;
use crate::prelude::*;

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    ReLu,
    Tanh,
    Softmax,
    Sigmoid,
}

impl Activation {
    pub fn activate<D: Dimension>(&self, inputs: &Tensor<D>) -> Tensor<D> {
        match self {
            Self::ReLu => inputs.mapv(|input| self.relu(input)),
            Self::Tanh => inputs.mapv(|input| self.tanh(input)),
            Self::Softmax => self.softmax(inputs),
            Self::Sigmoid => inputs.mapv(|input| self.sigmoid(input)),
        }
    }
}

impl<D> Layer<D, D> for Activation
where
    D: Dimension,
{
    fn forward(&self, outputs: &Tensor<D>) -> Tensor<D> {
        self.activate(outputs)
    }
}
