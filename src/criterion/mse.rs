use super::Criterion;
use crate::prelude::*;
use std::ops::Neg;

impl Criterion {
    pub fn mse<D, A>(&self, predicted: Tensor<D>, target: Array<A, D>) -> Tensor<D>
    where
        D: Dimension,
        A: Into<f64> + Neg<Output = A>,
        Array<A, D>: IntoIterator<Item = A>,
    {
        let dim = predicted.raw_dim();

        let mut mse = vec![];
        for (pred, actual) in predicted.into_iter().zip(target) {
            mse.push((pred - actual).powf(2.0));
        }

        Tensor::from_shape_vec(dim, mse).unwrap()
    }
}
