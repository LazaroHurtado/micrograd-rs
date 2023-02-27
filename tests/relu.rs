extern crate micrograd_rs;
use micrograd_rs::activation as Activation;
use micrograd_rs::prelude::*;
use micrograd_rs::Layer;

const VALUES: [f64; 5] = [-4.13, -0.0003123, 0.0, 0.00619, 4.32];

#[test]
fn valid_relu_activation() {
    let inputs = arr1(&VALUES).mapv(|val| Value::from(val));
    let relu = Activation::ReLU.forward(&inputs).into_raw_vec();
    let outputs = relu
        .iter()
        .map(|output| output.value())
        .collect::<Vec<f64>>();

    let actuals = [0.0, 0.0, 0.0, VALUES[3], VALUES[4]];

    assert_eq!(outputs, actuals);
}
