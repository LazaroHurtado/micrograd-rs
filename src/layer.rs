use super::neuron::{Activation, Neuron};
use super::value::Value;
use std::fmt;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, activation_fn: Option<Activation>) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin, activation_fn)).collect();

        Layer { neurons }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.parameters())
            .flatten()
            .collect()
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.call(input.clone()))
            .collect()
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, neuron) in self.neurons.iter().enumerate() {
            writeln!(f, "\t[Neuron-{:?}]\n{:?}", idx, neuron)?;
        }
        Ok(())
    }
}
