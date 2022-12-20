use super::value::Value;
use std::fmt;

pub trait Module: fmt::Debug {
    fn parameters(&self) -> Vec<Value> {
        vec![]
    }
    fn forward(&self, input: Vec<Value>) -> Vec<Value>;
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

    pub fn forward(&self, inputs: Vec<f64>) -> Vec<Value> {
        let input_values: Vec<Value> = inputs.into_iter().map(|input| input.into()).collect();
        self.layers
            .iter()
            .fold(input_values, |output, layer| layer.forward(output))
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
