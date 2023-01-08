#![allow(dead_code)]
use super::{Op, Value};

pub enum UnaryOps {
    Exp(Value),
    Log(Value),
    ReLu(Value),
}

impl Op for UnaryOps {
    fn into_inner(self) -> Vec<Value> {
        match self {
            Self::Exp(value) => vec![value],
            Self::Log(value) => vec![value],
            Self::ReLu(value) => vec![value],
        }
    }

    fn variables(&self) -> Vec<&Value> {
        match self {
            Self::Exp(value) => vec![value],
            Self::Log(value) => vec![value],
            Self::ReLu(value) => vec![value],
        }
    }

    fn propagate(&self, source: &Value) {
        let grad = &source.grad().unwrap();

        match self {
            Self::Exp(exponent) => {
                if exponent.should_compute_grad() {
                    *exponent.grad_mut() += grad * source
                }
            }
            Self::Log(variable) => {
                if variable.should_compute_grad() {
                    *variable.grad_mut() += grad / source
                }
            }
            Self::ReLu(unactivated) => {
                if unactivated.should_compute_grad() {
                    let one_if_greater_than_zero = source.value().ceil().min(1.0);

                    *unactivated.grad_mut() += grad * &one_if_greater_than_zero;
                }
            }
        };
    }
}
