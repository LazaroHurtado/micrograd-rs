use super::Activation;
use crate::value::Value;

impl Activation {
    pub fn sigmoid(&self, value: Value) -> Value {
        let one = Value::new(1.0);
        one / (one + (-&value).exp())
    }
}