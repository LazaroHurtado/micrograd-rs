use crate::{tensor::Tensor, Layer};
use ndarray::Dimension;

mod relu;
mod sigmoid;
mod softmax;
mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;

pub trait Activation<D: Dimension>: Layer<D, D> {
    fn activate(&self, inputs: &Tensor<D>) -> Tensor<D>;
}
