use super::Criterion;
use crate::prelude::*;
use crate::{Activation, Layer};

impl Criterion {
    pub fn cross_entropy<D>(&self, logits: &Tensor<D>, target: &Tensor<D>) -> Tensor<D>
    where
        D: Dimension,
    {
        let probabilities = Activation::Softmax.forward(logits).into_raw_vec();

        match target.shape() {
            [.., 1] => self.cross_entropy_with_class_indices(&probabilities, target),
            _ => self.cross_entropy_with_class_probs(&probabilities, target),
        }
    }

    pub fn cross_entropy_with_class_indices<D>(
        &self,
        probabilities: &[Value],
        class_indices: &Tensor<D>,
    ) -> Tensor<D>
    where
        D: Dimension,
    {
        let dim = class_indices.dim();

        let mut cross_entropy = vec![];
        for class_index in class_indices {
            cross_entropy.push(-probabilities[class_index.value() as usize].log());
        }

        Tensor::from_shape_vec(dim, cross_entropy).unwrap()
    }

    pub fn cross_entropy_with_class_probs<D>(
        &self,
        probabilities: &[Value],
        class_probabilities: &Tensor<D>,
    ) -> Tensor<D>
    where
        D: Dimension,
    {
        let mut dim = class_probabilities.raw_dim();
        dim[0] = 1;

        let cross_entropy = probabilities
            .iter()
            .zip(class_probabilities)
            .map(|(p, class_p)| &(-(p).log()) * class_p)
            .sum::<Value>();

        Tensor::from_shape_vec(dim, vec![cross_entropy]).unwrap()
    }
}
