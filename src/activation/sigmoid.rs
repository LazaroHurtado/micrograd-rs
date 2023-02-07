use super::Activation;
use crate::value::Value;

impl Activation {
    pub fn sigmoid(&self, value: Value) -> Value {
        Value::one() / (Value::one() + (-&value).exp())
    }
}
