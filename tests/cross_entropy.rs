extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use micrograd_rs::{Criterion, Reduction};

#[test]
fn valid_cross_entropy_loss_for_class_indices_with_mean_reduction() {
    let input = tensor![1.6832, -0.1644, -0.4215, -1.3357, -1.0669];
    let target = tensor!([1.], requires_grad = false);

    let criterion = Criterion::CrossEntropy;

    let actual_loss = 2.17853808;
    let loss = criterion.loss(Reduction::Mean, input, target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_cross_entropy_loss_for_class_probabilities_with_mean_reduction() {
    let input = tensor![1.6832, -0.1644, -0.4215, -1.3357, -1.0669];
    let target = tensor!([0.172, 0.321, 0.207, 0.021, 0.279], requires_grad = false);

    let criterion = Criterion::CrossEntropy;

    let actual_loss = 2.19036555;
    let loss = criterion.loss(Reduction::Mean, input, target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_batched_cross_entropy_loss_for_class_indices_with_sum_reduction() {
    let input = tensor![
        [24.47, -8.23, 0.82, -7.02],
        [-6.32, -7.01, 3.86, 4.51],
        [9.72, 7.84, -3.82, 1.27]
    ];
    let target = tensor!([[0.], [1.], [3.]], requires_grad = false);

    let criterion = Criterion::CrossEntropy;

    let actual_loss = 20.53227325;
    let loss = criterion.loss(Reduction::Sum, input, target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}

#[test]
fn valid_batched_cross_entropy_loss_for_class_probabilities_with_sum_reduction() {
    let input = tensor![
        [24.47, -8.23, 0.82, -7.02],
        [-6.32, -7.01, 3.86, 4.51],
        [9.72, 7.84, -3.82, 1.27]
    ];
    let target = tensor!(
        [
            [0.64, 0.12, 0.22, 0.01],
            [0.12, 0.73, 0.04, 0.11],
            [0.04, 0.06, 0.12, 0.78]
        ],
        requires_grad = false
    );

    let criterion = Criterion::CrossEntropy;

    let actual_loss = 28.06797328;
    let loss = criterion.loss(Reduction::Sum, input, target);

    assert_abs_diff_eq!(loss.value(), actual_loss, epsilon = 1e-6);
}
