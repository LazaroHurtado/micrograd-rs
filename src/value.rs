use super::operation::{Backpropagation, Op};
use std::{
    cell::RefCell,
    fmt,
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

#[derive(PartialEq)]
pub struct Data {
    pub value: f64,
    pub grad: f64,
    operation: Option<Op>,
    back_pass: bool,
}

#[derive(PartialEq)]
pub struct Value(pub Rc<RefCell<Data>>);

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl Value {
    pub fn new(value: f64) -> Self {
        let data = Data {
            value,
            grad: 0.0,
            operation: None,
            back_pass: false,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn with_op(value: f64, operation: Op) -> Self {
        let data = Data {
            value,
            grad: 0.0,
            operation: Some(operation),
            back_pass: false,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn value(&self) -> f64 {
        self.0.borrow().value
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn backward(&self) {
        let topo_order = self.topo_sort();
        self.0.borrow_mut().grad = 1.0;

        topo_order.iter().rev().for_each(|value| {
            let data = value.0.borrow();
            if let Some(op) = &data.operation {
                op.propagate(&data)
            }
        });
    }

    fn topo_sort(&self) -> Vec<Value> {
        let mut order = vec![];

        let mut data = self.0.borrow_mut();
        data.back_pass = true;

        if let Some(op) = &data.operation {
            for operand in op.equation() {
                if !operand.0.borrow().back_pass {
                    order.append(&mut operand.topo_sort());
                }
            }
        }
        order.push(self.clone());

        order
    }

    pub fn zero_grad(&self) {
        let mut data = self.0.borrow_mut();
        data.grad = 0.0;
        data.back_pass = false;
    }

    pub fn powf(&self, exponent: f64) -> Self {
        let value = self.value();
        Value::with_op(value, Op::Pow(self.clone(), exponent))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::new(value)
    }
}

impl Sum<Self> for Value {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Value::new(0.0), |sum, other| sum + other)
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        let value = self.value() + other.value();
        let operation = Op::Add(self.clone(), other.clone());

        Value::with_op(value, operation)
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Self::Output {
        let value = self.value() + other.value();
        let operation = Op::Add(self.clone(), other.clone());

        Value::with_op(value, operation)
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        // let value = self.value() - other.value();
        // let operation = Op::Sub(self.clone(), other.clone());
        //
        // Value::with_op(value, operation)
        self + (other * Value::new(-1.0))
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Self::Output {
        // let value = self.value() - other.value();
        // let operation = Op::Sub(self.clone(), other.clone());
        //
        // Value::with_op(value, operation)
        self + &(other * &Value::new(-1.0))
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        let value = self.value() * other.value();
        let operation = Op::Mul(self.clone(), other.clone());

        Value::with_op(value, operation)
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        let value = self.value() * other.value();
        let operation = Op::Mul(self.clone(), other.clone());

        Value::with_op(value, operation)
    }
}

impl Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        // let value = self.value() / other.value();
        // let operation = Op::Div(self.clone(), other.clone());
        //
        // Value::with_op(value, operation)
        self * other.powf(-1.0)
    }
}

impl Div<&Value> for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        // let value = self.value() / other.value();
        // let operation = Op::Div(self.clone(), other.clone());
        //
        // Value::with_op(value, operation)
        self * &other.powf(-1.0)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data: {}, grad: {})", self.value(), self.grad(),)
    }
}
