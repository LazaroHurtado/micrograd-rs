extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::{Filter, Module, Pooling};
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

    let pooled_input = max_pool_1d.forward(input);
    assert_eq!(
        pooled_input,
        tensor![[val!(1), val!(2)], [val!(4), val!(5)]]
    );
}

#[test]
fn valid_max_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let max_pool_2d = Pooling::Max(Filter::new(size, stride));

    let pooled_input = max_pool_2d.forward(input);
    assert_eq!(
        pooled_input,
        tensor![
            [[val!(4), val!(5)], [val!(10), val!(11)]],
            [[val!(16), val!(17)], [val!(22), val!(23)]],
            [[val!(28), val!(29)], [val!(34), val!(35)]]
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

    let pooled_input = max_pool_3d.forward(input);
    assert_eq!(
        pooled_input,
        Tensor::from_shape_vec(
            (3, 1, 2, 2),
            vec![
                val!(16),
                val!(17),
                val!(22),
                val!(23),
                val!(40),
                val!(41),
                val!(46),
                val!(47),
                val!(64),
                val!(65),
                val!(70),
                val!(71)
            ]
        )
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

    let pooled_input = avg_pool_1d.forward(input);
    assert_eq!(
        pooled_input,
        tensor![[val!(0.5), val!(1.5)], [val!(3.5), val!(4.5)]]
    );
}

#[test]
fn valid_avg_pooling_2d() {
    let (channels, height, width) = (3, 4, 3);
    let input = shaped_tensor_from_iter(0..36, (channels, height, width));

    let size = (2, 2);
    let stride = (2, 1);
    let avg_pool_2d = Pooling::Average(Filter::new(size, stride));

    let pooled_input = avg_pool_2d.forward(input);
    assert_eq!(
        pooled_input,
        tensor![
            [[val!(2), val!(3)], [val!(8), val!(9)]],
            [[val!(14), val!(15)], [val!(20), val!(21)]],
            [[val!(26), val!(27)], [val!(32), val!(33)]]
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

    let pooled_input = avg_pool_3d.forward(input);
    assert_eq!(
        pooled_input,
        Tensor::from_shape_vec(
            (3, 1, 2, 2),
            vec![
                val!(8),
                val!(9),
                val!(14),
                val!(15),
                val!(32),
                val!(33),
                val!(38),
                val!(39),
                val!(56),
                val!(57),
                val!(62),
                val!(63)
            ]
        )
        .unwrap()
    );
}
