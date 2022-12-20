use crate::mlp::Module;
use crate::value::Value;

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
    fn forward(&self, outputs: Vec<Value>) -> Vec<Value> {
        outputs
            .into_iter()
            .map(|neuron_output| self.activate(neuron_output))
            .collect()
    }
}
