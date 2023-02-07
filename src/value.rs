use super::ops::{BinaryOps, Op, Ops, UnaryOps};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};
use ordered_float::NotNan;

use std::cell::{RefCell, RefMut};
use std::f64::consts::E;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;
use std::{fmt, mem};

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

#[derive(Default)]
pub struct Data {
    pub value: NotNan<f64>,
    grad: Option<Value>,
    operation: Ops,
    back_pass: bool,
    requires_grad: bool,
}

pub struct Value(Rc<RefCell<Data>>);

impl Value {
    pub fn new(value: f64) -> Self {
        let data = Data {
            value: NotNan::new(value).expect("Value cannot be NaN"),
            requires_grad: true,
            ..Default::default()
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn with_op<T: Op + Into<Ops>>(value: f64, operation: T) -> Self {
        let data = Data {
            value: NotNan::new(value).expect("Value cannot be NaN"),
            operation: operation.into(),
            requires_grad: true,
            ..Default::default()
        };

        Value(Rc::new(RefCell::new(data)))
    }

    pub fn requires_grad(&mut self, requires: bool) {
        self.0.borrow_mut().requires_grad = requires;
    }

    pub fn should_compute_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn max(self, other: Value) -> Self {
        if self.value() > other.value() {
            self
        } else {
            other
        }
    }

    pub fn value(&self) -> f64 {
        self.0.borrow().value.into_inner()
    }

    pub fn value_mut(&self) -> RefMut<NotNan<f64>> {
        RefMut::map(self.0.borrow_mut(), |data| &mut data.value)
    }

    pub fn grad(&self) -> Option<Value> {
        self.0.borrow().grad.clone()
    }

    pub fn grad_mut(&self) -> RefMut<Value> {
        RefMut::map(self.0.borrow_mut(), |data| {
            data.grad.get_or_insert(Value::zero())
        })
    }

    pub fn zero_grad(&self) {
        let mut data = self.0.borrow_mut();
        data.grad = None;
        data.back_pass = false;
    }

    pub fn powf<T: Into<f64>>(&self, raw_exponent: T) -> Self {
        let mut exponent = Value::from(raw_exponent);
        exponent.requires_grad(false);

        let value = self.value().powf(exponent.value());
        Value::with_op(value, BinaryOps::Pow(self.clone(), exponent))
    }

    pub fn pow(&self, exponent: Self) -> Self {
        let value = self.value().powf(exponent.value());
        Value::with_op(value, BinaryOps::Pow(self.clone(), exponent))
    }

    pub fn exp(&self) -> Self {
        let value = E.powf(self.value());
        Value::with_op(value, UnaryOps::Exp(self.clone()))
    }

    pub fn log(&self) -> Self {
        let value = self.value().ln();
        Value::with_op(value, UnaryOps::Log(self.clone()))
    }

    pub fn backward(&self) {
        let topo_order = self.topo_sort();
        self.0.borrow_mut().grad = Some(Value::one());

        for source in topo_order.iter().rev() {
            let data = &source.0;
            data.borrow_mut().back_pass = false;
            data.borrow().operation.propagate(source);
        }
    }

    fn topo_sort(&self) -> Vec<Value> {
        let mut order = vec![];

        let mut data = self.0.borrow_mut();
        data.back_pass = true;
        data.grad = Some(Value::zero());

        let operation = &data.operation;
        for operand in operation.variables() {
            if !operand.0.borrow().back_pass {
                order.append(&mut operand.topo_sort());
            }
        }

        order.push(self.clone());
        order
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        if Rc::strong_count(&self.0) > 1 {
            return;
        }

        let refrences = |data: &Rc<RefCell<Data>>| {
            let grad_op = mem::take(&mut data.borrow_mut().grad);
            let mut refrences = if let Some(grad) = grad_op {
                vec![grad]
            } else {
                vec![]
            };

            let ops = mem::take(&mut data.borrow_mut().operation);
            let vars = ops.into_inner();
            refrences.extend(vars);
            refrences
        };
        let mut stack: Vec<Value> = refrences(&self.0);

        while let Some(mut curr) = stack.pop() {
            let data = mem::take(&mut curr.0);

            if Rc::strong_count(&data) == 1 {
                stack.extend(refrences(&data))
            }
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::zero()
    }
}

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

impl ScalarOperand for Value {}

impl Zero for Value {
    fn zero() -> Self {
        Value::from(0.0)
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
        Value::from(1.0)
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

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::zero() - self
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        &Value::zero() - self
    }
}

macro_rules! impl_binary_ops {
    ($trait: ident, $mth: ident, $operator: tt, $op_varient: tt) => {
        impl $trait<Value> for Value {
            type Output = Value;

            fn $mth(self, rhs: Self) -> Self::Output {
                let result = self.value() $operator rhs.value();
                let operation = BinaryOps::$op_varient(self, rhs);

                Value::with_op(result, operation)
            }
        }

        impl<'a> $trait<&'a Value> for &'a Value {
            type Output = Value;

            fn $mth(self, rhs: Self) -> Self::Output {
                let result = self.value() $operator rhs.value();
                let operation = BinaryOps::$op_varient(self.clone(), rhs.clone());

                Value::with_op(result, operation)
            }
        }

        impl<T> $trait<T> for Value where T: Into<f64> {
            type Output = Value;

            fn $mth(self, rhs: T) -> Self::Output {
                let mut rhs_val = Value::from(rhs);
                rhs_val.requires_grad(false);

                let value = self.value() $operator rhs_val.value();
                let operation = BinaryOps::$op_varient(self, rhs_val);

                Value::with_op(value, operation)
            }
        }

        impl<'a, T> $trait<&'a T> for &'a Value where T: Into<f64> + Copy {
            type Output = Value;

            fn $mth(self, rhs: &'a T) -> Self::Output {
                let mut rhs_val = Value::from(*rhs);
                rhs_val.requires_grad(false);

                let value = self.value() $operator rhs_val.value();
                let operation = BinaryOps::$op_varient(self.clone(), rhs_val);

                Value::with_op(value, operation)
            }
        }
    }
}

impl_binary_ops!(Add, add, +, Add);
impl_binary_ops!(Sub, sub, -, Sub);
impl_binary_ops!(Mul, mul, *, Mul);
impl_binary_ops!(Div, div, /, Div);

macro_rules! impl_binary_assign_ops {
    ($trait: ident, $mth: ident, $operator: tt) => {
        impl $trait for Value {
            fn $mth(&mut self, rhs: Self) {
                *self = self.clone() $operator rhs;
            }
        }

        impl $trait<&Value> for Value {
            fn $mth(&mut self, rhs: &Value) {
                *self = self.clone() $operator rhs.clone();
            }
        }
    }
}

impl_binary_assign_ops!(AddAssign, add_assign, +);
impl_binary_assign_ops!(SubAssign, sub_assign, -);
impl_binary_assign_ops!(MulAssign, mul_assign, *);
impl_binary_assign_ops!(DivAssign, div_assign, /);

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
