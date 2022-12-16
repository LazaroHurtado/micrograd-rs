use super::value::Value;
use num_traits::{Num, NumAssign, NumOps};
use std::ops::Neg;

pub trait Backpropagation<T> {
    fn backward(&self);
    fn propagate(&self, grad: T);
}

#[derive(Debug, Eq, PartialEq)]
pub enum Op<T> {
    Add(Value<T>, Value<T>),
    Sub(Value<T>, Value<T>),
    Mul(Value<T>, Value<T>),
    Div(Value<T>, Value<T>),
}

impl<T> Backpropagation<T> for Op<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    fn backward(&self) {
        let (lhs, rhs) = match self {
            Op::Add(lhs, rhs) => (lhs, rhs),
            Op::Sub(lhs, rhs) => (lhs, rhs),
            Op::Mul(lhs, rhs) => (lhs, rhs),
            Op::Div(lhs, rhs) => (lhs, rhs),
        };
        lhs.backward(false);
        rhs.backward(false);
    }

    fn propagate(&self, grad: T) {
        match self {
            Op::Add(Value(lhs), Value(rhs)) => {
                lhs.borrow_mut().grad += grad;
                rhs.borrow_mut().grad += grad;
            }
            Op::Sub(Value(lhs), Value(rhs)) => {
                lhs.borrow_mut().grad += grad;
                rhs.borrow_mut().grad += -grad;
            }
            Op::Mul(Value(lhs), Value(rhs)) => {
                let mut rhs_mut = rhs.borrow_mut();
                let mut lhs_mut = lhs.borrow_mut();

                lhs_mut.grad += rhs_mut.data * grad;
                rhs_mut.grad += lhs_mut.data * grad;
            }
            Op::Div(Value(lhs), Value(rhs)) => {
                let mut rhs_mut = rhs.borrow_mut();
                let mut lhs_mut = lhs.borrow_mut();

                lhs_mut.grad = grad / rhs_mut.data;
                rhs_mut.grad = (-lhs_mut.data * grad) / (rhs_mut.data * rhs_mut.data);
            }
        };

        self.backward();
    }
}
