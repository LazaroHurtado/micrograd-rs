use super::Activation;
use crate::value::Value;

impl Activation {
    pub fn tanh(&self, value: Value) -> Value {
        ((&value * &2.0).exp() - 1.0) / ((&value * &2.0).exp() + 1.0)
    }
}
