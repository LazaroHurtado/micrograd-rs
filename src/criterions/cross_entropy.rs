use std::marker::PhantomData;

use ndarray::RemoveAxis;

use super::{Criterion, Reduction};
use crate::activations as Activation;
use crate::prelude::*;
use crate::Layer;

pub struct CrossEntropy<D, E>(PhantomData<D>, PhantomData<E>);

impl<D> CrossEntropy<D, D>
where
    D: Dimension + RemoveAxis,
{
    fn with_class_probabilities(
        probabilities: &Tensor<D>,
        class_probabilities: &Tensor<D>,
    ) -> Tensor<D> {
        let mut dim = class_probabilities.raw_dim();
        dim[class_probabilities.ndim() - 1] = 1;

        let cross_entropy = match probabilities.ndim() {
            1 => vec![Self::normalized_with_prob(
                probabilities,
                class_probabilities,
            )],
            _ => {
                let mut normalized = vec![];

                for (batch_probs, batch_class) in probabilities
                    .outer_iter()
                    .zip(class_probabilities.outer_iter())
                {
                    let batch_probs = batch_probs.to_owned();
                    let batch_class = batch_class.to_owned();

                    let normalized_batch = Self::normalized_with_prob(&batch_probs, &batch_class);
                    normalized.push(normalized_batch);
                }

                normalized
            }
        };

        Tensor::from_shape_vec(dim, cross_entropy).unwrap()
    }

    fn normalized_with_prob<E: Dimension>(
        probabilities: &Tensor<E>,
        class_probabilities: &Tensor<E>,
    ) -> Value {
        probabilities
            .iter()
            .zip(class_probabilities)
            .map(|(p, class_p)| &(-(p).log()) * class_p)
            .sum::<Value>()
    }
}

impl<D, E> CrossEntropy<D, E>
where
    D: Dimension<Smaller = E> + RemoveAxis,
    E: Dimension<Larger = D>,
{
    fn with_class_indices(probabilities: &Tensor<D>, class_indices: &Tensor<E>) -> Tensor<E> {
        let indices = class_indices.clone().into_raw_vec();

        let cross_entropy = match probabilities.ndim() {
            1 => {
                let index = indices[0].value() as usize;
                let probabilities = probabilities.clone().into_raw_vec();

                vec![-probabilities[index].log()]
            }
            _ => {
                let mut normalized = vec![];

                for (i, batch_probs) in probabilities.outer_iter().enumerate() {
                    let index = indices[i].value() as usize;
                    let single_probs = batch_probs.to_owned();

                    let normalized_batch = -single_probs.into_raw_vec()[index].log();
                    normalized.push(normalized_batch);
                }

                normalized
            }
        };

        Tensor::from_shape_vec(class_indices.dim(), cross_entropy).unwrap()
    }
}

macro_rules! impl_criterion_for_class_indices {
    [$(($predicted_dim: ident, $target_dim: ident)),*] => {
        $(impl Criterion<$predicted_dim, $target_dim> for CrossEntropy<$predicted_dim, $target_dim> {
            fn loss(reduction: Reduction, predicted: &Tensor<$predicted_dim>, target: &Tensor<$target_dim>) -> Value {
                let n = predicted.ndim();
                let probabilities = Activation::Softmax(n-1).forward(predicted);
                let cross_entropy = Self::with_class_indices(&probabilities, target);

                Self::reduce(reduction, &cross_entropy)
            }
        })*
    };
}
impl_criterion_for_class_indices![(Ix1, Ix0), (Ix2, Ix1), (Ix3, Ix2)];

macro_rules! impl_criterion_for_class_probabilites {
    [$($dim: ident),*] => {
        $(impl Criterion<$dim, $dim> for CrossEntropy<$dim, $dim> {
            fn loss(reduction: Reduction, predicted: &Tensor<$dim>, target: &Tensor<$dim>) -> Value {
                let n = predicted.ndim();
                let probabilities = Activation::Softmax(n-1).forward(predicted);
                let cross_entropy = Self::with_class_probabilities(&probabilities, target);

                Self::reduce(reduction, &cross_entropy)
            }
        })*
    };
}
impl_criterion_for_class_probabilites![Ix1, Ix2, Ix3];
