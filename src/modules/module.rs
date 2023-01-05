use crate::prelude::*;

pub trait Module<D>
where
    D: Dimension,
{
    fn parameters(&self) -> Tensor<Ix1> {
        Tensor::from_vec(vec![])
    }

    fn forward(&self, input: Tensor<D>) -> Tensor<D>;
}
