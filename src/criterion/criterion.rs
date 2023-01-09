#![allow(clippy::module_inception)]

use crate::prelude::*;
use ndarray::RemoveAxis;

pub enum Reduction {
    Mean,
    Sum,
}

pub enum Criterion {
    MSE,
    CrossEntropy,
}

impl Criterion {
    pub fn loss<D>(&self, reduction: Reduction, predicted: &Tensor<D>, target: &Tensor<D>) -> Value
    where
        D: Dimension + RemoveAxis,
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

    pub fn element_wise_loss<D>(&self, predicted: &Tensor<D>, target: &Tensor<D>) -> Tensor<D>
    where
        D: Dimension + RemoveAxis,
    {
        match predicted.shape() {
            [_] => self.single_sample_loss(predicted, target),
            [_, ..] => self.batched_loss(predicted, target),
            _ => panic!("Unsupported shape"),
        }
    }

    fn single_sample_loss<D>(&self, predicted: &Tensor<D>, target: &Tensor<D>) -> Tensor<D>
    where
        D: Dimension,
    {
        match self {
            Self::MSE => self.mse(predicted, target),
            Self::CrossEntropy => self.cross_entropy(predicted, target),
        }
    }

    fn batched_loss<D>(&self, predicted_batch: &Tensor<D>, target_batch: &Tensor<D>) -> Tensor<D>
    where
        D: Dimension + RemoveAxis,
    {
        let mut dim = predicted_batch.raw_dim();
        let mut batched_loss = vec![];

        for (predicted, target) in predicted_batch.outer_iter().zip(target_batch.outer_iter()) {
            batched_loss.append(
                &mut self
                    .single_sample_loss(&predicted.to_owned(), &target.to_owned())
                    .into_raw_vec(),
            );
        }

        dim.set_last_elem(batched_loss.len() / dim[0]);
        Tensor::from_shape_vec(dim, batched_loss).unwrap()
    }
}
