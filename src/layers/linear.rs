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

impl Layer<Ix1, Ix1> for Linear {
    fn forward(&self, input: &Tensor<Ix1>) -> Tensor<Ix1> {
        input.dot(&self.weights) + &self.biases
    }

    fn weights(&self) -> Tensor<Ix1> {
        self.weights.clone().into_shape(self.weights.len()).unwrap()
    }

    fn biases(&self) -> Tensor<Ix1> {
        self.biases.clone()
    }

    fn set_weights(&self, new_weights: &[f64]) {
        for (v, &weight) in self.weights.iter().zip(new_weights) {
            *v.value_mut() = weight.into();
        }
    }

    fn set_biases(&self, new_biases: &[f64]) {
        for (v, &bias) in self.biases.iter().zip(new_biases) {
            *v.value_mut() = bias.into();
        }
    }

    fn is_trainable(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
