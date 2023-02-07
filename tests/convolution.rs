extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::{Conv1D, Conv2D, Conv3D, Layer};

#[test]
fn conv_returns_valid_parameter_count() {
    let (in_channels, out_channels) = (3, 13);

    let (n, m, k) = (3, 8, 4);
    let conv3d = Conv3D::new(
        in_channels,
        out_channels,
        (0, 0, 0),
        Filter::new((n, m, k), (1, 0, 0), (1, 1, 1)),
    );

    let parameters_per_kernel = in_channels * n * m * k;
    let total_kernel_parameters = out_channels * parameters_per_kernel;
    let total_parameters = total_kernel_parameters + out_channels; // one bias parameter per
                                                                   // kernel

    assert_eq!(conv3d.parameters().len(), total_parameters);
}

#[test]
fn valid_conv1d_padding() {
    let padding = 2;
    let (in_channels, out_channels) = (1, 1);

    let conv1d = Conv1D::new(in_channels, out_channels, padding, Filter::new(3, 2, 1));

    let input = tensor![[1., 1., 1.]];
    let padded_input = tensor![[0., 0., 1., 1., 1., 0., 0.]];

    assert_eq!(conv1d.pad_input(input), padded_input);
}

#[test]
fn valid_conv2d_padding() {
    let padding = (1, 2);
    let (in_channels, out_channels) = (1, 1);

    let conv2d = Conv2D::new(
        in_channels,
        out_channels,
        padding,
        Filter::new((2, 2), (1, 1), (1, 1)),
    );

    let input = tensor![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
    let padded_input = tensor![
        [
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 2., 0., 0.],
            [0., 0., 3., 4., 0., 0.],
            [0., 0., 0., 0., 0., 0.]
        ],
        [
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 5., 6., 0., 0.],
            [0., 0., 7., 8., 0., 0.],
            [0., 0., 0., 0., 0., 0.]
        ]
    ];

    assert_eq!(conv2d.pad_input(input), padded_input);
}
