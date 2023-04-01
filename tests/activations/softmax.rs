extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::activations as Activation;
use micrograd_rs::prelude::*;
use micrograd_rs::Layer;
use ndarray::arr3;

const VALUES_1D: [f64; 4] = [11.02, 10.26, 8.7, 12.4];
const VALUES_2D: [[f64; 4]; 2] = [VALUES_1D, [8.21, 9.1, 4.6, 7.67]];
const VALUES_3D: [[[f64; 4]; 2]; 2] = [
    VALUES_2D,
    [[3.82, 4.05, 0.21, 4.55], [5.61, 8.91, 10.41, 18.4]],
];

#[test]
fn valid_softmax_activation() {
    let inputs = arr1(&VALUES_1D).mapv(|val| Value::from(val));
    let softmaxed = Activation::Softmax(0).forward(&inputs).into_raw_vec();
    let outputs = softmaxed.iter().map(|output| output.value());

    let actuals = [0.18047813, 0.08440353, 0.01773622, 0.71738213];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_softmax_activation_for_2d() {
    let inputs = arr2(&VALUES_2D).mapv(|val| Value::from(val));
    let softmaxed = Activation::Softmax(1).forward(&inputs).into_raw_vec();
    let outputs = softmaxed.iter().map(|output| output.value());

    let actuals = [
        0.18047813, 0.08440353, 0.01773622, 0.71738213, 0.24722308, 0.60202025, 0.00668784,
        0.14406881,
    ];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_softmax_activation_for_2d_across_cols() {
    let inputs = arr2(&VALUES_2D).mapv(|val| Value::from(val));
    let softmaxed = Activation::Softmax(0).forward(&inputs).into_raw_vec();
    let outputs = softmaxed.iter().map(|output| output.value());

    let actuals = [
        0.94321381,
        0.761332714,
        0.98369750,
        0.99125075,
        0.05678618,
        0.23866728,
        0.01630249,
        0.00874924,
    ];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_softmax_activation_for_3d_across_depth() {
    let inputs = arr3(&VALUES_3D).mapv(|val| Value::from(val));
    let softmaxed = Activation::Softmax(0).forward(&inputs).into_raw_vec();
    let outputs = softmaxed.iter().map(|output| output.value());

    let actuals = [
        0.99925397, 0.99799479, 0.99979452, 0.99961039, 0.93086157, 0.54735761, 0.00298847,
        0.00002187, 0.00074602, 0.00200520, 0.00020547, 0.00038960, 0.06913842, 0.45264238,
        0.99701152, 0.99997812,
    ];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}
