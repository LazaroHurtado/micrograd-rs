use crate::prelude::*;
use std::ops::Neg;

pub enum Reduction {
    Mean,
    Sum,
}

pub enum Criterion {
    MSE,
}

impl Criterion {
    pub fn loss<D, A>(
        &self,
        reduction: Reduction,
        predicted: Tensor<D>,
        target: Array<A, D>,
    ) -> Value
    where
        D: Dimension,
        A: Into<f64> + Neg<Output = A>,
        Array<A, D>: IntoIterator<Item = A>,
    {
        let unreduced = self.element_wise_loss(predicted, target);

        match reduction {
            Reduction::Mean => {
                let n = unreduced.len() as f64;
                unreduced.sum() / n
            }
            Reduction::Sum => unreduced.sum(),
        }
    }

    pub fn element_wise_loss<D, A>(&self, predicted: Tensor<D>, target: Array<A, D>) -> Tensor<D>
    where
        D: Dimension,
        A: Into<f64> + Neg<Output = A>,
        Array<A, D>: IntoIterator<Item = A>,
    {
        match self {
            Self::MSE => self.mse(predicted, target),
        }
    }
}
