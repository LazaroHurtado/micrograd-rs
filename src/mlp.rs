use super::tensor::Tensor;
use super::value::Value;
use ndarray::{Dimension, Ix1, Ix2};
use std::fmt;

pub trait Module: fmt::Debug {
    fn parameters(&self) -> Vec<Value> {
        vec![]
    }
    fn forward(&self, input: Tensor<Ix1>) -> Tensor<Ix1>;
}

pub struct MLP {
    layers: Vec<Box<dyn Module>>,
}

impl MLP {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        MLP { layers }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    pub fn zero_grad(&self) {
        self.parameters().iter().for_each(|value| value.zero_grad());
    }

    pub fn forward(&self, inputs: Tensor<Ix1>) -> Tensor<Ix1> {
        self.layers
            .iter()
            .fold(inputs, |output, layer| layer.forward(output))
    }

    pub fn forward_batch<D: Dimension>(&self, batches: Tensor<Ix2>) -> Tensor<D> {
        let mut outputs = vec![];

        for batch in batches.rows() {
            let batch_output = self.forward(batch.to_owned());
            outputs.append(&mut batch_output.to_vec());
        }

        Tensor::from_vec(outputs)
            .into_dimensionality::<D>()
            .unwrap()
    }
}

impl fmt::Debug for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MLP: ")?;
        for (idx, layer) in self.layers.iter().enumerate() {
            writeln!(f, "[Layer-{:?}]\n{:?}", idx, layer)?;
        }
        Ok(())
    }
}
