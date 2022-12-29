use crate::prelude::*;

pub trait Module<D>
where
    D: Dimension,
{
    fn parameters(&self) -> Vec<Value> {
        vec![]
    }

    fn forward(&self, input: Tensor<D>) -> Tensor<D>;
}
