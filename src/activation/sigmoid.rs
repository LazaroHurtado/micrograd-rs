use ndarray::Dimension;

use super::Activation;
use crate::prelude::{One, Value};
use crate::tensor::Tensor;
use crate::Layer;

pub struct Sigmoid;

impl<D> Activation<D> for Sigmoid
where
    D: Dimension,
{
    fn activate(&self, unactivated: &Tensor<D>) -> Tensor<D> {
        unactivated.mapv(|value| Value::one() / (Value::one() + (-&value).exp()))
    }
}

impl<D> Layer<D, D> for Sigmoid
where
    D: Dimension,
{
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        self.activate(input)
    }
}
