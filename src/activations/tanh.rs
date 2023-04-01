use ndarray::Dimension;

use super::Activation;
use crate::{tensor::Tensor, Layer};

pub struct Tanh;

impl<D> Activation<D> for Tanh
where
    D: Dimension,
{
    fn activate(&self, unactivated: &Tensor<D>) -> Tensor<D> {
        unactivated.mapv(|value| ((&value * &2.0).exp() - 1.0) / ((&value * &2.0).exp() + 1.0))
    }
}

impl<D> Layer<D, D> for Tanh
where
    D: Dimension,
{
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        self.activate(input)
    }

    fn name(&self) -> String {
        String::from("Tanh")
    }
}
