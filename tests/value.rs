extern crate micrograd_rs;
use micrograd_rs::prelude::*;

fn nth_derivative(n: usize, x: Value, y: Value) -> f64 {
    (0..n)
        .fold(y, |d: Value, _| {
            x.zero_grad();
            d.backward();
            x.grad()
        })
        .value()
}

#[test]
fn first_derivative() {
    let x = Value::from(3.0);
    let y = x.powf(3.0);

    let actual_derivative = 3.0 * (3.0_f64).powf(2.0);
    assert_eq!(nth_derivative(1, x, y), actual_derivative);
}

#[test]
fn second_derivative() {
    let x = Value::from(3.0);
    let y = x.powf(3.0);

    let actual_derivative = 6.0 * 3.0;
    assert_eq!(nth_derivative(2, x, y), actual_derivative);
}

#[test]
fn third_derivative() {
    let x = Value::from(3.0);
    let y = x.powf(3.0);

    let actual_derivative = 6.0;
    assert_eq!(nth_derivative(3, x, y), actual_derivative);
}

#[test]
fn fourth_derivative() {
    let x = Value::from(3.0);
    let y = x.powf(3.0);

    let actual_derivative = 0.0;
    assert_eq!(nth_derivative(4, x, y), actual_derivative);
}
