extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use std::f64::consts::E;

fn nth_derivative(n: usize, x: Value, y: Value) -> f64 {
    if n == 0 {
        return y.value();
    };

    (0..n)
        .fold(y, |d: Value, _| {
            x.zero_grad();
            d.backward();
            x.grad().unwrap_or_else(Value::zero)
        })
        .value()
}

#[test]
fn valid_neg() {
    let x = Value::from(3.0);
    let x_neg = -&x;

    assert_eq!(x_neg.value(), -3.0);
    assert_eq!(x.value(), 3.0);
}

#[test]
fn valid_add_grads() {
    let x = Value::from(3.0);
    let y = &x + &4.0;

    let actual_derivatives = [7.0, 1.0, 0.0];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_eq!(
            nth_derivative(i, x.clone(), y.clone()),
            actual_derivatives[i]
        );
    }
}

#[test]
fn valid_add_assign_grads() {
    let mut x = Value::from(3.0);
    x += Value::from(4.0);

    assert_eq!(x.value(), 7.0);
}

#[test]
fn valid_sub_grads() {
    let x = Value::from(3.0);
    let y = &x - &5.0;

    let actual_derivatives = [-2.0, 1.0, 0.0];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_eq!(
            nth_derivative(i, x.clone(), y.clone()),
            actual_derivatives[i]
        );
    }
}

#[test]
fn valid_sub_assign_grads() {
    let mut x = Value::from(3.0);
    x -= Value::from(13.0);

    assert_eq!(x.value(), -10.0);
}

#[test]
fn valid_mul_grads() {
    let x = Value::from(3.0);
    let y = (&x * &2.0) + 3.0;

    let actual_derivatives = [9.0, 2.0, 0.0];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_eq!(
            nth_derivative(i, x.clone(), y.clone()),
            actual_derivatives[i]
        );
    }
}

#[test]
fn valid_mul_assign_grads() {
    let mut x = Value::from(5.0);
    x *= Value::from(-2.2);

    assert_eq!(x.value(), -11.0);
}

#[test]
fn valid_div_grads() {
    let x = Value::from(3.0);
    let y = (&x + &4.0) / ((&x * &2.0) + 3.0);

    let actual_derivatives = [(7.0 / 9.0), (-5.0 / 81.0), (20.0 / 729.0)];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_abs_diff_eq!(
            nth_derivative(i, x.clone(), y.clone()),
            actual_derivatives[i],
            epsilon = 1e-6
        );
    }
}

#[test]
fn valid_div_assign_grads() {
    let mut x = Value::from(27.0);
    x /= Value::from(9.0);

    assert_eq!(x.value(), 3.0);
}

#[test]
fn valid_pow_grads() {
    let x = Value::from(3.0);
    let y = x.powf(3.0);

    let actual_derivatives = [27.0, 27.0, 18.0, 6.0, 0.0];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_eq!(
            nth_derivative(i, x.clone(), y.clone()),
            actual_derivatives[i]
        );
    }
}

#[test]
fn valid_exp_grads() {
    let a = Value::from(3.0);
    let b = a.powf(2.0);
    let y = b.exp();

    let rep = E.powf(9.0);
    let actual_derivatives = [rep, 6.0 * rep, 2.0 * (rep + 18.0 * rep)];
    let n = actual_derivatives.len();

    for i in 0..n {
        assert_abs_diff_eq!(
            nth_derivative(i, a.clone(), y.clone()),
            actual_derivatives[i],
            epsilon = 1e-6
        );
    }
}
