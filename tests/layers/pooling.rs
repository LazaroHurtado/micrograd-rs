extern crate micrograd_rs;
use micrograd_rs::pooling::{AvgPool, MaxPool};
use micrograd_rs::prelude::*;
use micrograd_rs::Layer;
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
    let input = shaped_tensor_from_iter(0..6, (1, channels, width));

    let size = 2;
    let stride = 1;
    let max_pool_1d = MaxPool::new(size, stride, 0, 1);

    let pooled_input = max_pool_1d.forward(&input).into_raw_vec();
    assert_eq!(pooled_input, values![1., 2., 4., 5.]);
}

#[test]
fn valid_max_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (1, channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let max_pool_2d = MaxPool {
        size: Dim(size),
        stride: Dim(stride),
        ..Default::default()
    };

    let pooled_input = max_pool_2d.forward(&input).into_raw_vec();
    assert_eq!(
        pooled_input,
        values![4., 5., 10., 11., 16., 17., 22., 23., 28., 29., 34., 35.]
    );
}

#[test]
fn valid_max_pooling_3d() {
    let (channels, depth, height, width) = (3, 2, 4, 3);
    let input = shaped_tensor_from_iter(0..72, (1, channels, depth, height, width));

    let size = (2, 2, 2);
    let stride = (1, 2, 1);
    let max_pool_3d = MaxPool::new(size, stride, (0, 0, 0), (1, 1, 1));

    let pooled_input = max_pool_3d.forward(&input).into_raw_vec();
    assert_eq!(
        pooled_input,
        values![16., 17., 22., 23., 40., 41., 46., 47., 64., 65., 70., 71.]
    );
}

#[test]
fn valid_avg_pooling_1d() {
    let (channels, width) = (2, 3);
    let input = shaped_tensor_from_iter(0..6, (1, channels, width));

    let size = 2;
    let stride = 1;
    let avg_pool_1d = AvgPool {
        size: Dim(size),
        stride: Dim(stride),
        ..Default::default()
    };

    let pooled_input = avg_pool_1d.forward(&input).into_raw_vec();
    assert_eq!(pooled_input, values![0.5, 1.5, 3.5, 4.5]);
}

#[test]
fn valid_avg_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (1, channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let avg_pool_2d = AvgPool::new(size, stride, (0, 0), (1, 1));

    let pooled_input = avg_pool_2d.forward(&input).into_raw_vec();
    assert_eq!(
        pooled_input,
        values![2., 3., 8., 9., 14., 15., 20., 21., 26., 27., 32., 33.]
    );
}

#[test]
fn valid_avg_pooling_3d() {
    let (channels, depth, height, width) = (3, 2, 4, 3);
    let input = shaped_tensor_from_iter(0..72, (1, channels, depth, height, width));

    let size = (2, 2, 2);
    let stride = (1, 2, 1);
    let avg_pool_3d = AvgPool {
        size: Dim(size),
        stride: Dim(stride),
        ..Default::default()
    };

    let pooled_input = avg_pool_3d.forward(&input).into_raw_vec();
    assert_eq!(
        pooled_input,
        values![8., 9., 14., 15., 32., 33., 38., 39., 56., 57., 62., 63.]
    );
}
