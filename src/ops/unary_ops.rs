use super::{Op, Value};

pub enum UnaryOps {
    Exp(Value),
    Log(Value),
    ReLu(Value),
    NoOp,
}

impl Op for UnaryOps {
    fn variables(&self) -> Vec<Value> {
        match self {
            Self::Exp(value) => vec![value.clone()],
            Self::Log(value) => vec![value.clone()],
            Self::ReLu(value) => vec![value.clone()],
            Self::NoOp => vec![],
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
            Self::NoOp => (),
        };
    }
}
