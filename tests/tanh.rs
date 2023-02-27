extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::activation as Activation;
use micrograd_rs::prelude::*;
use micrograd_rs::Layer;

const VALUES: [f64; 6] = [-24.71, -0.73, -0.12, 71.23, 0.41, 0.0];

#[test]
fn valid_tanh_activation() {
    let inputs = arr1(&VALUES).mapv(|val| Value::from(val));
    let tanh = Activation::Tanh.forward(&inputs).into_raw_vec();
    let outputs = tanh.iter().map(|output| output.value());

    let actuals = [-1.0, -0.62306535, -0.11942729, 1.0, 0.38847268, 0.0];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}
