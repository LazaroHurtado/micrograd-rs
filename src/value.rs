use super::operation::{Backpropagation, Op};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};
use std::{
    cell::RefCell,
    fmt,
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[macro_export]
macro_rules! values {
    [$($x: expr),*] => {{
        let mut values = vec![];

        $(
            values.push(Value::from($x));
        )*
        values
    }};
}

#[macro_export]
macro_rules! val {
    ($x: expr) => {
        Value::from($x)
    };
}

pub struct Data {
    pub value: f64,
    grad: Option<Value>,
    operation: Option<Op>,
    back_pass: bool,
}

impl Data {
    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn grad(&mut self) -> Value {
        self.grad_mut().clone()
    }

    pub fn grad_mut(&mut self) -> &mut Value {
        self.grad.get_or_insert(Value::new(0.0))
    }
}

pub struct Value(pub Rc<RefCell<Data>>);

impl Value {
    pub fn new(value: f64) -> Self {
        let data = Data {
            value,
            grad: None,
            operation: None,
            back_pass: false,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn with_op(value: f64, operation: Op) -> Self {
        let data = Data {
            value,
            grad: None,
            operation: Some(operation),
            back_pass: false,
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn max(self, other: Value) -> Self {
        if self.value() > other.value() {
            self
        } else {
            other
        }
    }

    pub fn zero() -> Self {
        Value::from(0.0)
    }

    pub fn one() -> Self {
        Value::from(1.0)
    }

    pub fn value(&self) -> f64 {
        self.0.borrow().value()
    }

    pub fn grad(&self) -> Self {
        let mut data = self.0.borrow_mut();
        data.grad()
    }

    pub fn backward(&self) {
        let topo_order = self.topo_sort();
        self.0.borrow_mut().grad = Some(Value::new(1.0));

        topo_order.iter().rev().for_each(|source| {
            let mut data = source.0.borrow_mut();
            let value = data.value();
            let grad = data.grad();

            if let Some(op) = &data.operation {
                op.propagate(&value, &grad)
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
        data.grad = None;
        data.back_pass = false;
    }

    pub fn powf(&self, exponent: f64) -> Self {
        let value = self.value().powf(exponent);
        Value::with_op(value, Op::Pow(self.clone(), exponent))
    }
}

impl ScalarOperand for Value {}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Zero for Value {
    fn zero() -> Self {
        Value::zero()
    }

    fn set_zero(&mut self) {
        *self = Value::zero()
    }

    fn is_zero(&self) -> bool {
        *self == Value::zero()
    }
}

impl One for Value {
    fn one() -> Self {
        Value::one()
    }

    fn set_one(&mut self) {
        *self = Value::one()
    }

    fn is_one(&self) -> bool {
        *self == Value::one()
    }
}

impl<T> From<T> for Value
where
    T: Into<f64>,
{
    fn from(value: T) -> Self {
        Value::new(value.into())
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

impl AddAssign for Value {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl AddAssign<&Value> for Value {
    fn add_assign(&mut self, other: &Value) {
        *self = self.clone() + other.clone();
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.0.borrow_mut().value *= -1.0;
        self
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl Add<Value> for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.value() + rhs.value();
        let operation = Op::Add(self, rhs);

        Value::with_op(value, operation)
    }
}

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        let value = self.value() + rhs.value();
        let operation = Op::Add(self.clone(), rhs.clone());

        Value::with_op(value, operation)
    }
}

impl<T> Add<T> for Value
where
    T: Into<f64>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let rhs = Value::from(other);
        let value = self.value() + rhs.value();
        let operation = Op::Add(self, rhs);

        Value::with_op(value, operation)
    }
}

impl<T> Add<&T> for &Value
where
    T: Into<f64> + Copy,
{
    type Output = Value;

    fn add(self, other: &T) -> Self::Output {
        let rhs = Value::from(*other);
        let value = self.value() + rhs.value();
        let operation = Op::Add(self.clone(), rhs);

        Value::with_op(value, operation)
    }
}

impl<T> Sub<T> for Value
where
    Self: Add<T, Output = Self>,
    T: Neg<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a, T> Sub<&'a T> for &'a Value
where
    Value: Add<T, Output = Value>,
    &'a T: Neg<Output = T>,
{
    type Output = Value;

    fn sub(self, rhs: &'a T) -> Self::Output {
        self.clone() + (-rhs)
    }
}

impl Mul<Value> for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let value = self.value() * rhs.value();
        let operation = Op::Mul(self, rhs);

        Value::with_op(value, operation)
    }
}

impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        let value = self.value() * rhs.value();
        let operation = Op::Mul(self.clone(), rhs.clone());

        Value::with_op(value, operation)
    }
}

impl<T> Mul<T> for Value
where
    T: Into<f64>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        let rhs = Value::from(other);
        let value = self.value() * rhs.value();
        let operation = Op::Mul(self, rhs);

        Value::with_op(value, operation)
    }
}

impl<T> Mul<&T> for &Value
where
    T: Into<f64> + Copy,
{
    type Output = Value;

    fn mul(self, other: &T) -> Self::Output {
        let rhs = Value::from(*other);
        let value = self.value() * rhs.value();
        let operation = Op::Mul(self.clone(), rhs);

        Value::with_op(value, operation)
    }
}

impl Div<Value> for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

impl Div<&Value> for &Value {
    type Output = Value;

    fn div(self, rhs: &Value) -> Self::Output {
        self * &rhs.powf(-1.0)
    }
}

impl<T> Div<T> for Value
where
    T: Into<f64>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        let rhs = Value::from(other);
        self * rhs.powf(-1.0)
    }
}

impl<T> Div<&T> for &Value
where
    T: Into<f64> + Copy,
{
    type Output = Value;

    fn div(self, other: &T) -> Self::Output {
        let rhs = Value::from(*other);
        self * &rhs.powf(-1.0)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value(data: {}, grad: {:?})",
            self.value(),
            self.0.borrow().grad,
        )
    }
}
