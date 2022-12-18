use super::value::{Data, Value};

pub trait Backpropagation {
    fn propagate(&self, data: &Data);
}

#[derive(Debug, PartialEq)]
pub enum Op {
    Add(Value, Value),
    // Sub(Value, Value),
    Mul(Value, Value),
    // Div(Value, Value),
    Pow(Value, f64),
    ReLu(Value),
    TanH(Value),
}

impl Op {
    pub fn equation(&self) -> Vec<Value> {
        match self {
            Op::Add(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            // Op::Sub(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Mul(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            // Op::Div(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Pow(value, _) => vec![value.clone()],
            Op::ReLu(value) => vec![value.clone()],
            Op::TanH(value) => vec![value.clone()],
        }
    }
}

impl Backpropagation for Op {
    fn propagate(&self, data: &Data) {
        let value = data.value;
        let grad = data.grad;

        match self {
            Op::Add(Value(lhs), Value(rhs)) => {
                lhs.borrow_mut().grad += grad;
                rhs.borrow_mut().grad += grad;
            }
            // Op::Sub(Value(lhs), Value(rhs)) => {
            //     lhs.borrow_mut().grad += grad;
            //     rhs.borrow_mut().grad += -grad;
            // }
            Op::Mul(Value(lhs), Value(rhs)) => {
                let mut rhs_mut = rhs.borrow_mut();
                let mut lhs_mut = lhs.borrow_mut();

                lhs_mut.grad += rhs_mut.value * grad;
                rhs_mut.grad += lhs_mut.value * grad;
            }
            // Op::Div(Value(lhs), Value(rhs)) => {
            //     let mut rhs_mut = rhs.borrow_mut();
            //     let mut lhs_mut = lhs.borrow_mut();
            //
            //     lhs_mut.grad += grad / rhs_mut.value;
            //     rhs_mut.grad += (-lhs_mut.value * grad) / (rhs_mut.value.powf(2.0));
            // }
            Op::Pow(Value(variable), exponent) => {
                let mut variable_mut = variable.borrow_mut();

                variable_mut.grad += exponent * (variable_mut.value.powf(exponent - 1.0)) * grad;
            }
            Op::ReLu(Value(unactivated)) => {
                let one_if_greater_than_zero = value.ceil().min(1.0);
                unactivated.borrow_mut().grad += one_if_greater_than_zero * grad;
            }
            Op::TanH(Value(unactivated)) => {
                unactivated.borrow_mut().grad += (1.0 - value.powf(2.0)) * grad;
            }
        };
    }
}
