use crate::prelude::*;

pub trait Module {
    type Dim;

    fn parameters(&self) -> Vec<Value> {
        vec![]
    }

    fn forward(&self, input: Tensor<Self::Dim>) -> Tensor<Self::Dim>;
}
