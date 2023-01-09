use super::Activation;
use crate::prelude::*;

impl Activation {
    pub fn softmax<D: Dimension>(&self, inputs: &Tensor<D>) -> Tensor<D> {
        let inputs_dim = inputs.dim();

        let logits = inputs.clone().into_raw_vec();
        let max_logit = logits.clone().into_iter().reduce(Value::max).unwrap();
        let exp_sum = logits
            .iter()
            .map(|logit| (logit - &max_logit).exp())
            .sum::<Value>();

        let outputs = logits
            .iter()
            .map(|logit| &(logit - &max_logit).exp() / &exp_sum)
            .collect::<Vec<Value>>();

        Tensor::from_shape_vec(inputs_dim, outputs).unwrap()
    }
}
