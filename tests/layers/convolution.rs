extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use micrograd_rs::{Conv2D, Conv3D, Layer};

#[test]
fn conv_returns_valid_parameter_count() {
    let name = "conv3d";
    let (in_channels, out_channels) = (3, 13);

    let (n, m, k) = (3, 8, 4);
    let conv3d = Conv3D::new(
        name,
        in_channels,
        out_channels,
        (n, m, k),
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 1),
    );

    let parameters_per_kernel = in_channels * n * m * k;
    let total_kernel_parameters = out_channels * parameters_per_kernel;
    let total_parameters = total_kernel_parameters + out_channels; // one bias parameter per
                                                                   // kernel

    assert_eq!(conv3d.parameters().len(), total_parameters);
}

#[test]
fn valid_convolution() {
    let mut conv2d = Conv2D::new(1, 1, 1, (2, 2), (0, 0), (1, 1), (1, 1));
    conv2d.weights =
        Array4::from_shape_vec((1, 1, 2, 2), values![0.3954, -0.1740, -0.1890, 0.4909]).unwrap();
    conv2d.biases = Tensor::from_vec(values!(-0.1188));

    let input = Array4::from_shape_vec(
        (1, 1, 5, 4),
        values![
            -1.5237, 0.9591, -2.0597, 0.8249, -0.4506, -0.6975, 1.0153, -0.2838, -0.5344, -0.5019,
            -0.4378, 0.3062, 0.0597, 1.4820, 0.4158, 1.4295, 0.0612, -0.4898, -0.2115, -0.4827
        ],
    )
    .unwrap();

    let outputs = conv2d.forward(&input).mapv(|v| v.value()).into_raw_vec();
    let actuals = vec![
        -1.14539373,
        1.24905421,
        -1.40794710,
        -0.32098335,
        -0.69131062,
        0.56508859,
        0.47345934,
        -0.31705584,
        0.27797043,
        -0.60507223,
        0.38358044,
        -0.40010961,
    ];

    for (output, actual) in outputs.into_iter().zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}
