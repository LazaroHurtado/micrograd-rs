mod cross_entropy;
mod mse;

use crate::tensor::Tensor;
use crate::value::Value;
use ndarray::Dimension;

pub use cross_entropy::CrossEntropy;
pub use mse::MSE;

pub enum Reduction {
    Mean,
    Sum,
}

pub trait Criterion<D: Dimension, E: Dimension> {
    fn loss(reduction: Reduction, predicted: &Tensor<D>, target: &Tensor<E>) -> Value;

    fn reduce(reduction: Reduction, loss: &Tensor<E>) -> Value {
        match reduction {
            Reduction::Mean => {
                let n = loss.len() as f64;
                loss.sum() / n
            }
            Reduction::Sum => loss.sum(),
        }
    }
}
