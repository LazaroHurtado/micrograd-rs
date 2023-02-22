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

    fn name(&self) -> String {
        String::from("NO NAME")
    }

    fn weights(&self) -> (Vec<f64>, Vec<f64>) {
        let default_shape: Vec<f64> = vec![0.0];
        let default_weights: Vec<f64> = vec![0.0];
        (default_shape, default_weights)
    }

    fn biases(&self) -> Vec<f64> {
        let default_weights: Vec<f64> = vec![0.0];
        default_weights
    }
}
