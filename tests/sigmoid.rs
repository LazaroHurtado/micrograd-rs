extern crate micrograd_rs;
use micrograd_rs::activation as Activation;
use micrograd_rs::prelude::*;
use micrograd_rs::Layer;

const VALUES: [f64; 5] = [-4.13, -0.0003123, 0.0, 1.0, 4.32];

#[test]
fn valid_sigmoid_activation() {
    let inputs = arr1(&VALUES).mapv(|val| Value::from(val));
    let sigmoid = Activation::Sigmoid.forward(&inputs).into_raw_vec();
    let outputs = sigmoid
        .iter()
        .map(|output| output.value())
        .collect::<Vec<f64>>();

    let actuals = [
        0.015828313967089846,
        0.4999219250006346,
        0.5,
        0.7310585786300049,
        0.9868746816628972,
    ];

    assert_eq!(outputs, actuals);
}
