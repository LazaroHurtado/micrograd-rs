use super::value::Value;

pub trait Backpropagation {
    fn propagate(&self, value: &f64, grad: &Value);
}

pub enum Op {
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),
    Pow(Value, f64),
    Exp(Value, f64),
    ReLu(Value),
}

impl Op {
    pub fn equation(&self) -> Vec<Value> {
        match self {
            Op::Add(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Sub(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Mul(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
            Op::Div(numer, denom) => vec![numer.clone(), denom.clone()],
            Op::Pow(value, _) => vec![value.clone()],
            Op::Exp(value, _) => vec![value.clone()],
            Op::ReLu(value) => vec![value.clone()],
        }
    }
}

impl Backpropagation for Op {
    fn propagate(&self, value: &f64, grad: &Value) {
        match self {
            Op::Add(Value(lhs), Value(rhs)) => {
                let mut lhs_data = lhs.borrow_mut();
                *lhs_data.grad_mut() += grad;

                let mut rhs_data = rhs.borrow_mut();
                *rhs_data.grad_mut() += grad;
            }
            Op::Sub(Value(lhs), Value(rhs)) => {
                let mut lhs_data = lhs.borrow_mut();
                *lhs_data.grad_mut() += grad;

                let mut rhs_data = rhs.borrow_mut();
                *rhs_data.grad_mut() += -grad;
            }
            Op::Mul(lhs, rhs) => {
                *lhs.0.borrow_mut().grad_mut() += grad * rhs;
                *rhs.0.borrow_mut().grad_mut() += grad * lhs;
            }
            Op::Div(numer, denom) => {
                *numer.0.borrow_mut().grad_mut() += grad / denom;
                *denom.0.borrow_mut().grad_mut() += grad * &(numer / &denom.powf(2.0));
            }
            Op::Pow(variable, exponent) => {
                let derivative = &variable.powf(exponent - 1.0) * exponent;

                *variable.0.borrow_mut().grad_mut() += grad * &derivative;
            }
            Op::Exp(exponent, result) => *exponent.0.borrow_mut().grad_mut() += grad * result,
            Op::ReLu(Value(unactivated)) => {
                let mut unactivated_data = unactivated.borrow_mut();
                let one_if_greater_than_zero = value.ceil().min(1.0);

                *unactivated_data.grad_mut() += grad * &one_if_greater_than_zero;
            }
        };
    }
}
