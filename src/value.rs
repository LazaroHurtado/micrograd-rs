use super::operation::{Backpropagation, Op};
use num_traits::{Num, NumAssign, NumOps};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Debug, Eq, PartialEq)]
pub struct Data<T> {
    pub data: T,
    pub grad: T,
    operation: Option<Op<T>>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Value<T>(pub Rc<RefCell<Data<T>>>);

impl<T> Clone for Value<T> {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl<T> Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    pub fn new(data: T) -> Self {
        let data = Data {
            data,
            grad: 0.into(),
            operation: None,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn with_op(data: T, operation: Option<Op<T>>) -> Self {
        let data = Data {
            data,
            grad: 0.into(),
            operation,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn backward(&self, initial: bool) {
        if initial {
            self.0.borrow_mut().grad = 1.into();
        }

        let data = self.0.borrow();

        let grad = data.grad.clone();
        if let Some(op) = &data.operation {
            op.propagate(grad);
        }
    }
}

// impl<T: Add<Output = T> + From<u8> + Clone> From<T> for Value<T> {
//     fn from(x: T) -> Self {
//         Value {
//             data: x,
//             grad: 0.into(),
//             operation: None,
//         }
//     }
// }

impl<T> Add<Value<T>> for Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn add(self, other: Value<T>) -> Self::Output {
        let data = self.0.borrow().data + other.0.borrow().data;
        let operation = Some(Op::Add(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Add<&Value<T>> for &Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn add(self, other: &Value<T>) -> Self::Output {
        let data = self.0.borrow().data + other.0.borrow().data;
        let operation = Some(Op::Add(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Sub<Value<T>> for Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn sub(self, other: Value<T>) -> Self::Output {
        let data = self.0.borrow().data - other.0.borrow().data;
        let operation = Some(Op::Sub(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Sub<&Value<T>> for &Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn sub(self, other: &Value<T>) -> Self::Output {
        let data = self.0.borrow().data - other.0.borrow().data;
        let operation = Some(Op::Sub(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Mul<Value<T>> for Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn mul(self, other: Value<T>) -> Self::Output {
        let data = self.0.borrow().data * other.0.borrow().data;
        let operation = Some(Op::Mul(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Mul<&Value<T>> for &Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn mul(self, other: &Value<T>) -> Self::Output {
        let data = self.0.borrow().data * other.0.borrow().data;
        let operation = Some(Op::Mul(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Div<Value<T>> for Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn div(self, other: Value<T>) -> Self::Output {
        let data = self.0.borrow().data / other.0.borrow().data;
        let operation = Some(Op::Div(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}

impl<T> Div<&Value<T>> for &Value<T>
where
    T: Num + NumAssign + NumOps + Neg<Output = T> + From<u8> + Copy,
{
    type Output = Value<T>;

    fn div(self, other: &Value<T>) -> Self::Output {
        let data = self.0.borrow().data / other.0.borrow().data;
        let operation = Some(Op::Div(self.clone(), other.clone()));

        Value::with_op(data, operation)
    }
}
