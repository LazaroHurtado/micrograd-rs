extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use micrograd_rs::{Activation, Layer};

const VALUES: [f64; 4] = [11.02, 10.26, 8.7, 12.4];

#[test]
fn valid_softmax_activation() {
    let inputs = arr1(&VALUES).mapv(|val| Value::from(val));
    let softmaxed = Activation::Softmax.forward(&inputs).into_raw_vec();
    let outputs = softmaxed.iter().map(|output| output.value());

    let actuals = [0.18047813, 0.08440353, 0.01773622, 0.71738213];

    for (output, actual) in outputs.zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}
