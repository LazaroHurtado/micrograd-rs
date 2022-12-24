use super::utils::WeightInit;
use crate::prelude::*;
use std::fmt;

pub struct Neuron {
    weights: Tensor<Ix1>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize, nout: usize) -> Self {
        let weights =
            Tensor::from_shape_simple_fn(nin, || WeightInit::GlorotUniform.sample([nin, nout]));
        let bias = Value::zero();

        Neuron { weights, bias }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.to_vec();
        params.push(self.bias.clone());

        params
    }

    pub fn call(&self, inputs: Tensor<Ix1>) -> Value {
        let sum: Value = inputs.dot(&self.weights);
        let output = &sum + &self.bias;

        output
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, weight) in self.weights.iter().enumerate() {
            writeln!(f, "\t\t[Weight-{:?}]   {:?}", idx, weight)?;
        }
        Ok(())
    }
}
