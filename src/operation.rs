use super::value::Value;

pub trait Backpropagation {
    fn propagate(&self, source: &Value);
}

pub enum Op {
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    Pow(Value, Value),
    Exp(Value),
    Log(Value),
    ReLu(Value),
    NoOp,
}

impl Op {
    pub fn equation(&self) -> Vec<Value> {
        match self {
            Op::Add(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Sub(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Mul(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Div(numer, denom) => vec![numer.clone(), denom.clone()],
            Op::Pow(value, _) => vec![value.clone()],
            Op::Exp(value) => vec![value.clone()],
            Op::Log(value) => vec![value.clone()],
            Op::ReLu(value) => vec![value.clone()],
            Op::NoOp => vec![],
        }
    }
}

impl Backpropagation for Op {
    fn propagate(&self, source: &Value) {
        let grad = &source.grad().unwrap();

        match self {
            Op::Add(lhs, rhs) => {
                if lhs.should_compute_grad() {
                    *lhs.grad_mut() += grad;
                }

                if rhs.should_compute_grad() {
                    *rhs.grad_mut() += grad;
                }
            }
            Op::Sub(lhs, rhs) => {
                if lhs.should_compute_grad() {
                    *lhs.grad_mut() += grad;
                }

                if rhs.should_compute_grad() {
                    *rhs.grad_mut() += -grad;
                }
            }
            Op::Mul(lhs, rhs) => {
                if lhs.should_compute_grad() {
                    *lhs.grad_mut() += grad * rhs;
                }
                if rhs.should_compute_grad() {
                    *rhs.grad_mut() += grad * lhs;
                }
            }
            Op::Div(numer, denom) => {
                if numer.should_compute_grad() {
                    *numer.grad_mut() += grad / denom;
                }
                if denom.should_compute_grad() {
                    let derivative = -(numer / &denom.powf(2.0));
                    *denom.grad_mut() += grad * &derivative;
                }
            }
            Op::Pow(variable, exponent) => {
                if variable.should_compute_grad() {
                    let wrt_variable = &variable.pow(exponent - &1.0) * exponent;
                    *variable.grad_mut() += grad * &wrt_variable;
                }
                if exponent.should_compute_grad() {
                    let wrt_exponent = variable.log() * variable.pow(exponent.clone());
                    *exponent.grad_mut() += grad * &wrt_exponent;
                }
            }
            Op::Exp(exponent) => {
                if exponent.should_compute_grad() {
                    *exponent.grad_mut() += grad * source
                }
            }
            Op::Log(variable) => {
                if variable.should_compute_grad() {
                    *variable.grad_mut() += grad / source
                }
            }
            Op::ReLu(unactivated) => {
                if unactivated.should_compute_grad() {
                    let one_if_greater_than_zero = source.value().ceil().min(1.0);

                    *unactivated.grad_mut() += grad * &one_if_greater_than_zero;
                }
            }
            _ => (),
        };
    }
}
