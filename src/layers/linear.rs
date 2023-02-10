use super::Layer;
use crate::prelude::*;
use crate::utils::WeightInit;

pub struct Linear {
    weights: Tensor<Ix2>,
    biases: Tensor<Ix1>,
}

impl Linear {
    pub fn new(nin: usize, nout: usize) -> Self {
        let weights = Tensor::from_shape_simple_fn((nout, nin), || {
            WeightInit::GlorotUniform.sample([nin, nout])
        });
        let biases = Tensor::from_shape_simple_fn(nout, Value::zero);

        Linear { weights, biases }
    }
}

impl Layer<Ix1, Ix1> for Linear {
    fn parameters(&self) -> Tensor<Ix1> {
        let mut params = self.weights.clone().into_raw_vec();
        params.append(&mut self.biases.clone().into_raw_vec());

        Tensor::from_vec(params)
    }

    fn forward(&self, input: &Tensor<Ix1>) -> Tensor<Ix1> {
        let weights_t = self.weights.t().into_owned();

        input.dot(&weights_t) + &self.biases
    }
}
