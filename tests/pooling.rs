extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::{Module, Pooling};
use ndarray::IntoDimension;
use std::ops::Range;

fn shaped_tensor_from_iter<E, D: IntoDimension<Dim = E>>(iter: Range<u8>, shape: D) -> Tensor<E> {
    Tensor::from_iter(iter.map(|x| val!(x)))
        .into_shape(shape)
        .unwrap()
}

#[test]
fn valid_max_pooling_1d() {
    let (channels, width) = (2, 3);
    let input = shaped_tensor_from_iter(0..6, (channels, width));

    let size = 2;
    let stride = 1;
    let max_pool_1d = Pooling::Max(Filter::new(size, stride));

    let pooled_input = max_pool_1d.forward(&input);
    assert_eq!(pooled_input, tensor![[1., 2.], [4., 5.]]);
}

#[test]
fn valid_max_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let max_pool_2d = Pooling::Max(Filter::new(size, stride));

    let pooled_input = max_pool_2d.forward(&input);
    assert_eq!(
        pooled_input,
        tensor![
            [[4., 5.], [10., 11.]],
            [[16., 17.], [22., 23.]],
            [[28., 29.], [34., 35.]]
        ]
    );
}

#[test]
fn valid_max_pooling_3d() {
    let (channels, depth, height, width) = (3, 2, 4, 3);
    let input = shaped_tensor_from_iter(0..72, (channels, depth, height, width));

    let size = (2, 2, 2);
    let stride = (1, 2, 1);
    let max_pool_3d = Pooling::Max(Filter::new(size, stride));

    let pooled_input = max_pool_3d.forward(&input);
    assert_eq!(
        pooled_input,
        tensor![16., 17., 22., 23., 40., 41., 46., 47., 64., 65., 70., 71.]
            .into_shape((3, 1, 2, 2))
            .unwrap()
    );
}

#[test]
fn valid_avg_pooling_1d() {
    let (channels, width) = (2, 3);
    let input = shaped_tensor_from_iter(0..6, (channels, width));

    let size = 2;
    let stride = 1;
    let avg_pool_1d = Pooling::Average(Filter::new(size, stride));

    let pooled_input = avg_pool_1d.forward(&input);
    assert_eq!(pooled_input, tensor![[0.5, 1.5], [3.5, 4.5]]);
}

#[test]
fn valid_avg_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let avg_pool_2d = Pooling::Average(Filter::new(size, stride));

    let pooled_input = avg_pool_2d.forward(&input);
    assert_eq!(
        pooled_input,
        tensor![
            [[2., 3.], [8., 9.]],
            [[14., 15.], [20., 21.]],
            [[26., 27.], [32., 33.]]
        ]
    );
}

#[test]
fn valid_avg_pooling_3d() {
    let (channels, depth, height, width) = (3, 2, 4, 3);
    let input = shaped_tensor_from_iter(0..72, (channels, depth, height, width));

    let size = (2, 2, 2);
    let stride = (1, 2, 1);
    let avg_pool_3d = Pooling::Average(Filter::new(size, stride));

    let pooled_input = avg_pool_3d.forward(&input);
    assert_eq!(
        pooled_input,
        tensor![8., 9., 14., 15., 32., 33., 38., 39., 56., 57., 62., 63.]
            .into_shape((3, 1, 2, 2))
            .unwrap()
    );
}
