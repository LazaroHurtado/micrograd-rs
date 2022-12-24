use crate::mlp::Module;
use crate::prelude::*;

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    ReLu,
    TanH,
}

impl Activation {
    pub fn activate(&self, value: Value) -> Value {
        match self {
            Self::ReLu => self.relu(value),
            Self::TanH => self.tanh(value),
        }
    }
}

impl Module for Activation {
    fn forward(&self, outputs: Tensor<Ix1>) -> Tensor<Ix1> {
        outputs.map(|neuron_output| self.activate(neuron_output.clone()))
    }
}
