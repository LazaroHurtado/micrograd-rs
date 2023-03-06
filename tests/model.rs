extern crate micrograd_rs;
use micrograd_rs::activation as Activation;
use micrograd_rs::prelude::*;
use micrograd_rs::Conv2D;
use micrograd_rs::Pooling;
use micrograd_rs::{Layer, Linear, Model, Sequential};
use std::fs;

#[test]
fn valid_save_and_load_state_dict_for_linear_model() {
    let model1 = sequential!(
        Ix1,
        [
            Linear::new("fc1", 15, 20),
            Activation::Tanh,
            Linear::new("fc2", 20, 10),
            Activation::Tanh,
            Linear::new("fc3", 10, 1),
            Activation::Tanh
        ]
    );

    let mut model2 = sequential!(
        Ix1,
        [
            Linear::new("fc1", 15, 20),
            Activation::Tanh,
            Linear::new("fc2", 20, 10),
            Activation::Tanh,
            Linear::new("fc3", 10, 1),
            Activation::Tanh
        ]
    );

    assert_ne!(model1.state_dict(), model2.state_dict());

    let path = "linear.pickle";
    model1.save_state_dict(path);
    model2.load_state_dict(path);

    assert_eq!(model1.state_dict(), model2.state_dict());

    assert!(
        fs::remove_file(path).is_ok(),
        "File \"linear.pickle\" could not be removed."
    );
}

#[test]
fn valid_save_and_load_state_dict_for_convolutional_model() {
    let name = "conv2d";
    let padding = (1, 2);
    let (in_channels, out_channels) = (3, 1);

    let pool_size = (2, 2);
    let pool_stride = (2, 1);

    let model1 = sequential!(
        Ix3,
        [
            Conv2D::new(
                name,
                in_channels,
                out_channels,
                padding,
                Filter::new((2, 2), (1, 1), (1, 1)),
            ),
            Pooling::Average(Filter::new(pool_size, pool_stride, (1, 1))),
            Activation::Sigmoid
        ]
    );

    let mut model2 = sequential!(
        Ix3,
        [
            Conv2D::new(
                name,
                in_channels,
                out_channels,
                padding,
                Filter::new((2, 2), (1, 1), (1, 1)),
            ),
            Pooling::Average(Filter::new(pool_size, pool_stride, (1, 1))),
            Activation::Sigmoid
        ]
    );

    assert_ne!(model1.state_dict(), model2.state_dict());

    let path = "convolutional.pickle";
    model1.save_state_dict(path);
    model2.load_state_dict(path);

    assert_eq!(model1.state_dict(), model2.state_dict());

    assert!(
        fs::remove_file(path).is_ok(),
        "File \"convolutional.pickle\" could not be removed."
    );
}
