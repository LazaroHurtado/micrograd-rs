use super::mlp::Module;
use super::neuron::Neuron;
use super::value::Value;
use std::fmt;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();

        Layer { neurons }
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.parameters())
            .flatten()
            .collect()
    }

    fn forward(&self, input: Vec<Value>) -> Vec<Value> {
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
