use super::Criterion;
use crate::prelude::*;

impl Criterion {
    pub fn mse<D>(&self, predicted: &Tensor<D>, target: &Tensor<D>) -> Tensor<D>
    where
        D: Dimension,
    {
        let dim = predicted.raw_dim();

        let mut mse = vec![];
        for (pred, actual) in predicted.into_iter().zip(target) {
            mse.push((pred - actual).powf(2.0));
        }

        Tensor::from_shape_vec(dim, mse).unwrap()
    }
}
