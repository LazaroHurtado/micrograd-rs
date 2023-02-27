use ndarray::Dimension;

use super::Activation;
use crate::ops::UnaryOps;
use crate::tensor::Tensor;
use crate::value::Value;
use crate::Layer;

pub struct ReLU;

impl<D> Activation<D> for ReLU
where
    D: Dimension,
{
    fn activate(&self, unactivated: &Tensor<D>) -> Tensor<D> {
        unactivated.mapv(|value| {
            let activated = value.value().max(0.0);
            Value::with_op(activated, UnaryOps::ReLU(value))
        })
    }
}

impl<D> Layer<D, D> for ReLU
where
    D: Dimension,
{
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        self.activate(input)
    }
}
