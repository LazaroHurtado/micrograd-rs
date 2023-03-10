use crate::prelude::*;

pub trait Layer<In, Out>
where
    In: Dimension,
    Out: Dimension,
{
    fn forward(&self, input: &Tensor<In>) -> Tensor<Out>;

    fn parameters(&self) -> Tensor<Ix1> {
        let mut params = self.weights().into_raw_vec();
        params.append(&mut self.biases().into_raw_vec());

        Tensor::from_vec(params)
    }

    fn weights(&self) -> Tensor<Ix1> {
        Tensor::from_vec(vec![])
    }

    fn biases(&self) -> Tensor<Ix1> {
        Tensor::from_vec(vec![])
    }

    fn set_weights(&self, new_weights: &[f64]) {
        for (v, &weight) in self.weights().iter().zip(new_weights) {
            *v.value_mut() = weight.into();
        }
    }

    fn set_biases(&self, new_biases: &[f64]) {
        for (v, &bias) in self.biases().iter().zip(new_biases) {
            *v.value_mut() = bias.into();
        }
    }

    fn is_trainable(&self) -> bool {
        false
    }

    fn name(&self) -> String;
}
