use super::value::Value;
use std::ops::{AddAssign, Div, Mul, Neg};

pub trait Backpropagation<T> {
    fn propagate(&mut self, grad: T);
}

#[derive(Debug, Eq, PartialEq)]
pub enum Op<'a, T> {
    Add(&'a mut Value<T>, &'a mut Value<'a, T>),
    Sub(&'a mut Value<'a, T>, &'a mut Value<'a, T>),
    Mul(&'a mut Value<'a, T>, &'a mut Value<'a, T>),
    Div(&'a mut Value<'a, T>, &'a mut Value<'a, T>),
}

impl<T> Backpropagation<T> for Op<'_, T>
where
    T: AddAssign + Mul<Output = T> + Div<Output = T> + Neg<Output = T> + From<u8> + Copy,
{
    fn propagate(&mut self, grad: T) {
        let (lhs, rhs) = match self {
            Op::Add(lhs, rhs) => {
                lhs.grad += grad;
                rhs.grad += grad;
                (lhs, rhs)
            }
            Op::Sub(lhs, rhs) => {
                lhs.grad += grad;
                rhs.grad += -grad;
                (lhs, rhs)
            }
            Op::Mul(lhs, rhs) => {
                lhs.grad += rhs.data * grad;
                rhs.grad += lhs.data * grad;
                (lhs, rhs)
            }
            Op::Div(lhs, rhs) => {
                lhs.grad = grad / rhs.data;
                rhs.grad = (-lhs.data * grad) / (rhs.data * rhs.data);
                (lhs, rhs)
            }
        };

        lhs.backward(false);
        rhs.backward(false);
    }
}
