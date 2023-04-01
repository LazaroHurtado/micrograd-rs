use super::{Criterion, Reduction};
use crate::prelude::*;

pub struct MSE;

impl MSE {
    fn mse<D>(predicted: &Tensor<D>, target: &Tensor<D>) -> Tensor<D>
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

impl<D> Criterion<D, D> for MSE
where
    D: Dimension,
{
    fn loss(reduction: Reduction, predicted: &Tensor<D>, target: &Tensor<D>) -> Value {
        let mse = Self::mse(predicted, target);
        Self::reduce(reduction, &mse)
    }
}
