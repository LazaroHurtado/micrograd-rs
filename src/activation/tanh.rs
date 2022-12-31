use super::Activation;
use crate::operation::Op;
use crate::value::Value;
use std::f64::consts::E;

impl Activation {
    pub fn tanh(&self, value: Value) -> Value {
        let activated = (E.powf(2.0 * value.value()) - 1.0) / (E.powf(2.0 * value.value()) + 1.0);
        Value::with_op(activated, Op::Tanh(value))
    }
}
