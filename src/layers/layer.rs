use crate::prelude::*;

pub trait Layer<In, Out>
where
    In: Dimension,
    Out: Dimension,
{
    fn parameters(&self) -> Tensor<Ix1> {
        Tensor::from_vec(vec![])
    }

    fn forward(&self, input: &Tensor<In>) -> Tensor<Out>;
}
