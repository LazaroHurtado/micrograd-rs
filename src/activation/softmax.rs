use super::Activation;
use crate::operation::Op;
use crate::prelude::*;

impl Activation {
    pub fn softmax<D: Dimension>(&self, inputs: Tensor<D>) -> Tensor<D> {
        let n = inputs.len();
        let inputs_dim = inputs.dim();

        let input_values = inputs.into_raw_vec();
        let logits = input_values.iter().map(|logit| logit.value());
        let max_logit = logits.reduce(f64::max).unwrap();

        let exp_sum = input_values
            .iter()
            .map(|val| (val.value() - max_logit).exp())
            .sum::<f64>();
        let raw_outputs = input_values
            .iter()
            .map(|val| (val.value() - max_logit).exp() / exp_sum)
            .collect::<Vec<f64>>();

        let mut jacobian = vec![vec![0.0; n]; n];
        for idx in 0..n * n {
            let (i, j) = (idx / n, idx % n);

            jacobian[i][j] = if i == j {
                raw_outputs[i] * (1. - raw_outputs[i])
            } else {
                -raw_outputs[i] * raw_outputs[j]
            };
        }

        let outputs = raw_outputs
            .into_iter()
            .enumerate()
            .map(|(idx, raw_output)| {
                Value::with_op(
                    raw_output,
                    Op::Softmax(input_values.clone(), jacobian[idx].clone()),
                )
            })
            .collect::<Vec<Value>>();

        Tensor::from_shape_vec(inputs_dim, outputs).unwrap()
    }
}
