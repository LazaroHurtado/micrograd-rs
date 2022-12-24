use super::tensor::{DotProd, Tensor};
use super::value::Value;
use ndarray::Ix1;
use rand::distributions::{Distribution, Uniform};
use std::fmt;

pub struct Neuron {
    weights: Tensor<Ix1>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let uniform = Uniform::from(-1.0..=1.0);
        let mut rng = rand::thread_rng();

        let weights: Tensor<Ix1> = Tensor::from(
            (0..nin)
                .map(|_| {
                    let sample = uniform.sample(&mut rng);
                    Value::new(sample)
                })
                .collect::<Vec<Value>>(),
        );

        let sample = uniform.sample(&mut rng);
        let bias = Value::new(sample);

        Neuron { weights, bias }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.bias.clone()];
        params.append(&mut self.weights.to_vec());

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
