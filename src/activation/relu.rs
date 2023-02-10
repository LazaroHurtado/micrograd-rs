use super::Activation;
use crate::ops::UnaryOps;
use crate::value::Value;

impl Activation {
    pub fn relu(&self, value: Value) -> Value {
        let activated = value.value().max(0.0);
        Value::with_op(activated, UnaryOps::ReLU(value))
    }
}
