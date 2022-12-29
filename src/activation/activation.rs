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

impl<D> Module<D> for Activation
where
    D: Dimension,
{
    fn forward(&self, outputs: Tensor<D>) -> Tensor<D> {
        outputs.mapv(|neuron_output| self.activate(neuron_output.clone()))
    }
}
