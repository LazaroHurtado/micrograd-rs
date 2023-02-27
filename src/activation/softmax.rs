use ndarray::RemoveAxis;

use super::Activation;
use crate::{prelude::*, Layer};

pub struct Softmax(pub usize);

impl Softmax {
    fn softmax(&self, logits: &[Value]) -> Vec<Value> {
        let max_logit = logits.iter().cloned().reduce(Value::max).unwrap();
        let exp_sum = logits
            .iter()
            .map(|logit| (logit - &max_logit).exp())
            .sum::<Value>();

        logits
            .iter()
            .map(|logit| &(logit - &max_logit).exp() / &exp_sum)
            .collect::<Vec<Value>>()
    }

    fn dimension_factor<D: Dimension>(&self, inputs: &Tensor<D>) -> usize {
        let n = inputs.ndim();
        let shape = inputs.shape();

        shape.to_vec().iter().rev().take(n - self.0 - 1).product()
    }

    fn ordered_by_dimension<D: Dimension>(&self, inputs: &Tensor<D>) -> Vec<Value> {
        let dimension_factor = self.dimension_factor(inputs);

        let mut inputs: Vec<(usize, Value)> = inputs
            .clone()
            .into_raw_vec()
            .into_iter()
            .enumerate()
            .collect();
        inputs.sort_by_key(|(i, _)| i % dimension_factor);

        inputs.into_iter().map(|(_, v)| v).collect()
    }
}

impl<D> Activation<D> for Softmax
where
    D: Dimension + RemoveAxis,
{
    fn activate(&self, inputs: &Tensor<D>) -> Tensor<D> {
        let dim = inputs.raw_dim();

        let shape = inputs.shape();
        let chunk_size = shape[self.0];

        let dimension_factor = self.dimension_factor(inputs);
        let offset = if dimension_factor == 1 { chunk_size } else { 1 };

        let mut inputs = self.ordered_by_dimension(inputs);
        let mut outputs = vec![Value::zero(); inputs.len()];

        for (i, input) in inputs.chunks_exact_mut(chunk_size).enumerate() {
            let softmaxed_chunk = self.softmax(input);

            for (j, softmax) in softmaxed_chunk.into_iter().enumerate() {
                let proper_location = (j * dimension_factor) + (i * offset);
                outputs[proper_location] = softmax;
            }
        }

        Tensor::from_shape_vec(dim, outputs).unwrap()
    }
}

impl<D> Layer<D, D> for Softmax
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        self.activate(input)
    }
}
