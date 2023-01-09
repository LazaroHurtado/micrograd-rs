extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use micrograd_rs::{Criterion, Reduction};

#[test]
fn valid_mse_loss_with_mean_reduction() {
    let input = tensor![1.78, -2.02, -0.14, -2.0, -7.24];
    let target = tensor!([2.4, -1.86, 0.04, -1.5, -10.0], requires_grad = false);

    let criterion = Criterion::MSE;

    let actual_loss = 1.66200042;
    let loss = criterion.loss(Reduction::Mean, &input, &target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_mse_loss_with_sum_reduction() {
    let input = tensor![1.78, -2.02, -0.14, -2.0, -7.24];
    let target = tensor!([2.4, -1.86, 0.04, -1.5, -10.0], requires_grad = false);

    let criterion = Criterion::MSE;

    let actual_loss = 8.31;
    let loss = criterion.loss(Reduction::Sum, &input, &target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_batched_mse_loss_with_mean_reduction() {
    let input = tensor![
        [1.78, -2.02, -0.14, -2.0, -7.24],
        [3.41, -2.71, 12.43, -7.3, -9.82]
    ];
    let target = tensor!(
        [
            [2.4, -1.86, 0.04, -1.5, -10.0],
            [4.23, -3.2, 10.72, -4.87, -10.32]
        ],
        requires_grad = false
    );

    let criterion = Criterion::MSE;

    let actual_loss = 1.83015025;
    let loss = criterion.loss(Reduction::Mean, &input, &target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_batched_mse_loss_with_sum_reduction() {
    let input = tensor![
        [1.78, -2.02, -0.14, -2.0, -7.24],
        [3.41, -2.71, 12.43, -7.3, -9.82]
    ];
    let target = tensor!(
        [
            [2.4, -1.86, 0.04, -1.5, -10.0],
            [4.23, -3.2, 10.72, -4.87, -10.32]
        ],
        requires_grad = false
    );

    let criterion = Criterion::MSE;

    let actual_loss = 18.30150000;
    let loss = criterion.loss(Reduction::Sum, &input, &target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}
