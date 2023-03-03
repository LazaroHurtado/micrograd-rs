extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::prelude::*;
use micrograd_rs::BatchNorm;
use micrograd_rs::Layer;
use ndarray::RemoveAxis;

fn batch_norm_output<D: Dimension + RemoveAxis>(features: usize, input: &Tensor<D>) -> Vec<f64>
where
    D::Smaller: Dimension<Larger = D>,
{
    let batch_norm = BatchNorm::new("bn", features);
    batch_norm.forward(input).mapv(|v| v.value()).into_raw_vec()
}

#[test]
fn valid_1d_batch_norm() {
    let mini_batch = tensor![[[1., 2., 3.], [4., 5., 6.]]];

    let outputs = batch_norm_output(2, &mini_batch);
    let actuals = [-1.2247356, 0.0, 1.2247356, -1.2247357, 0.0, 1.2247355];

    for (output, actual) in outputs.into_iter().zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_2d_batch_norm() {
    let mini_batch = Tensor::<Ix4>::from_shape_vec(
        [1, 2, 2, 3],
        values![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    )
    .unwrap();

    let outputs = batch_norm_output(2, &mini_batch);
    let actuals = [
        -1.46384759,
        -0.87830855,
        -0.29276951,
        0.292769519,
        0.878308559,
        1.463847599,
        -1.46384759,
        -0.87830855,
        -0.29276951,
        0.292769519,
        0.878308559,
        1.463847599,
    ];

    for (output, actual) in outputs.into_iter().zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}

#[test]
fn valid_3d_batch_norm() {
    let mini_batch =
        Tensor::<Ix5>::from_shape_vec([1, 1, 2, 2, 2], values![1., 2., 3., 4., 5., 6., 7., 8.])
            .unwrap();

    let outputs = batch_norm_output(1, &mini_batch);
    let actuals = [
        -1.52752388,
        -1.09108853,
        -0.65465307,
        -0.21821773,
        0.21821764,
        0.65465301,
        1.09108841,
        1.52752376,
    ];

    for (output, actual) in outputs.into_iter().zip(actuals) {
        assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
    }
}
