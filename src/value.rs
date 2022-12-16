use super::operation::{Backpropagation, Op};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Eq, PartialEq)]
pub struct Value<'a, T> {
    pub data: T,
    pub grad: T,
    operation: Option<Op<'a, T>>,
}

impl<T> Value<'_, T>
where
    T: Mul<Output = T> + Div<Output = T> + Neg<Output = T> + AddAssign + From<u8> + Copy,
{
    pub fn new(data: T) -> Self {
        Value {
            data,
            grad: 0.into(),
            operation: None,
        }
    }

    pub fn backward(&mut self, initial: bool) {
        if initial {
            self.grad = 1.into();
        }

        if let Some(ref mut op) = self.operation {
            op.propagate(self.grad.clone());
        }
    }

    pub fn params(&mut self) -> Vec<&'_ mut Value<'_, T>> {
        let mut params = vec![self];

        if let Some(ref mut op) = self.operation {
            let (lhs, rhs) = match op {
                Op::Add(lhs, rhs) => (lhs, rhs),
                Op::Sub(lhs, rhs) => (lhs, rhs),
                Op::Mul(lhs, rhs) => (lhs, rhs),
                Op::Div(lhs, rhs) => (lhs, rhs),
            };

            params.append(&mut lhs.params());
            params.append(&mut rhs.params());
        }
        params
    }
}

impl<T: Add<Output = T> + From<u8> + Clone> From<T> for Value<'_, T> {
    fn from(x: T) -> Self {
        Value {
            data: x,
            grad: 0.into(),
            operation: None,
        }
    }
}

impl<'a, T: Add<Output = T> + From<u8> + Copy> Add<&'a mut Value<'a, T>> for &'a mut Value<'a, T> {
    type Output = Value<'a, T>;

    fn add(self, other: &'a mut Value<'a, T>) -> Self::Output {
        let data = self.data + other.data;

        let operation = Some(Op::Add(self, other));

        Value {
            data,
            grad: 0.into(),
            operation,
        }
    }
}

impl<'a, T: Sub<Output = T> + From<u8> + Copy> Sub<&'a mut Value<'a, T>> for &'a mut Value<'a, T> {
    type Output = Value<'a, T>;

    fn sub(self, other: &'a mut Value<'a, T>) -> Self::Output {
        let data = self.data - other.data;

        let operation = Some(Op::Sub(self, other));

        Value {
            data,
            grad: 0.into(),
            operation,
        }
    }
}

impl<'a, T: Mul<Output = T> + From<u8> + Copy> Mul<&'a mut Value<'a, T>> for &'a mut Value<'a, T> {
    type Output = Value<'a, T>;

    fn mul(self, other: &'a mut Value<'a, T>) -> Self::Output {
        let data = self.data * other.data;

        let operation = Some(Op::Mul(self, other));

        Value {
            data,
            grad: 0.into(),
            operation,
        }
    }
}

impl<'a, T: Div<Output = T> + From<u8> + Copy> Div<&'a mut Value<'a, T>> for &'a mut Value<'a, T> {
    type Output = Value<'a, T>;

    fn div(self, other: &'a mut Value<'a, T>) -> Self::Output {
        let data = self.data / other.data;

        let operation = Some(Op::Div(self, other));

        Value {
            data,
            grad: 0.into(),
            operation,
        }
    }
}
