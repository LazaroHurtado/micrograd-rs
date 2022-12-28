use crate::modules::Module;
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
    type Dim = Ix1;

    fn forward(&self, outputs: Tensor<Self::Dim>) -> Tensor<Self::Dim> {
        outputs.map(|neuron_output| self.activate(neuron_output.clone()))
    }
}
