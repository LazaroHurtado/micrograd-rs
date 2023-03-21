use std::ops::Add;

use super::Layer;
use crate::prelude::*;
use crate::utils::WeightInit;

pub struct Linear {
    name: String,
    weights: Tensor<Ix2>,
    biases: Tensor<Ix1>,
}

impl Linear {
    pub fn new(name: impl ToString, nin: usize, nout: usize) -> Self {
        let name = name.to_string();
        let weights = Tensor::from_shape_simple_fn((nin, nout), || {
            WeightInit::GlorotUniform.sample([nin, nout])
        });
        let biases = Tensor::from_shape_simple_fn(nout, Value::zero);

        Linear {
            name,
            weights,
            biases,
        }
    }
}

impl<D, E> Layer<D, D> for Linear
where
    D: Dimension,
    Tensor<D>: DotProd<Tensor<Ix2>, Output = Tensor<E>>,
    Tensor<E>: Add<Tensor<Ix1>, Output = Tensor<D>>,
{
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        input.dot(&self.weights) + self.biases.clone()
    }

    fn weights(&self) -> Tensor<Ix1> {
        self.weights.clone().into_shape(self.weights.len()).unwrap()
    }

    fn biases(&self) -> Tensor<Ix1> {
        self.biases.clone()
    }

    fn is_trainable(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
