use std::cell::Cell;

use ndarray::RemoveAxis;

use crate::{prelude::*, Layer};

pub struct BatchNorm {
    pub name: String,
    pub features: usize,
    pub eps: f64,
    pub momentum: f64,
    weight: Tensor<Ix1>,
    bias: Tensor<Ix1>,
    running_mean: Cell<Tensor<Ix1>>,
    running_var: Cell<Tensor<Ix1>>,
}

impl Default for BatchNorm {
    fn default() -> Self {
        BatchNorm {
            name: "Batch Normalization".into(),
            features: 0,
            eps: 1e-5,
            momentum: 0.1,
            weight: Tensor::ones(0),
            bias: Tensor::zeros(0),
            running_mean: Cell::new(Tensor::ones(0)),
            running_var: Cell::new(Tensor::ones(0)),
        }
    }
}

impl BatchNorm {
    pub fn new(name: impl ToString, features: usize) -> Self {
        Self {
            name: name.to_string(),
            features,
            weight: Tensor::ones(features),
            bias: Tensor::zeros(features),
            running_mean: Cell::new(Tensor::zeros(features)),
            running_var: Cell::new(Tensor::ones(features)),
            ..Default::default()
        }
    }

    fn update_running_stat<D: Dimension>(
        &self,
        celled_running_stat: &Cell<Tensor<Ix1>>,
        batch_stat: &Tensor<D>,
    ) {
        let scaled_batch_stat = (batch_stat * self.momentum)
            .into_shape(batch_stat.len())
            .unwrap();

        let running_stat = celled_running_stat.take();
        let scaled_running_stat = &running_stat * (1. - self.momentum);

        let new_running_stat = scaled_batch_stat + scaled_running_stat;
        celled_running_stat.set(new_running_stat);
    }

    fn iterative_mean<D: Dimension + RemoveAxis>(
        &self,
        input: &Tensor<D>,
        axes: &[usize],
    ) -> Tensor<D>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        axes.iter()
            .fold(input.clone(), |meaned, axis| self.mean::<D>(&meaned, *axis))
    }

    fn mean<D: Dimension + RemoveAxis>(&self, input: &Tensor<D>, axis: usize) -> Tensor<D>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        let mean = input.mean_axis(Axis(axis)).unwrap();
        mean.insert_axis(Axis(axis))
    }

    fn iterative_var<D: Dimension + RemoveAxis>(
        &self,
        input: &Tensor<D>,
        axes: &[usize],
        corrected: bool,
    ) -> Tensor<D>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        let shape = input.shape();
        let biased = axes.iter().map(|axis| shape[*axis] as f64).product::<f64>();

        let mean = self.iterative_mean(input, axes);
        let normalized = (input - mean).mapv(|v| v.powf(2.0));

        let var = self.iterative_mean(&normalized, axes);
        match corrected {
            true => (var * biased) / (biased - 1.0),
            false => var,
        }
    }

    fn reshape<D: Dimension>(&self, parameter: &Tensor<Ix1>, shape: &[usize]) -> Tensor<D> {
        let reshaped = parameter.clone().into_shape(shape).unwrap();
        reshaped.into_dimensionality::<D>().unwrap()
    }

    fn transform<D: Dimension>(&self, output: Tensor<D>) -> Tensor<D> {
        let mut reshape = vec![1; output.ndim()];
        reshape[0] = self.features;

        let broadcasted_weights = self.reshape::<D>(&self.weight, &reshape);
        let broadcasted_biases = self.reshape::<D>(&self.bias, &reshape);

        (output * broadcasted_weights) - broadcasted_biases
    }
}

impl<D> Layer<D, D> for BatchNorm
where
    D: Dimension + RemoveAxis,
    D::Smaller: Dimension<Larger = D>,
{
    fn parameters(&self) -> Tensor<Ix1> {
        let mut parameters = self.weight.clone().into_raw_vec();
        parameters.append(&mut self.bias.clone().into_raw_vec());

        Tensor::from_vec(parameters)
    }

    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        let n = input.ndim();

        let mut single_channel_mean_dims = (2..n).collect::<Vec<usize>>();
        let mut stat_dims = vec![0];
        stat_dims.append(&mut single_channel_mean_dims);

        let mean = self.iterative_mean(input, &stat_dims);

        let unbiased_var = self.iterative_var(input, &stat_dims, false);
        let biased_var = self.iterative_var(input, &stat_dims, true);

        let shifted_var = unbiased_var.mapv(|v| (v + self.eps).sqrt());

        let normalized = (input - &mean) / &shifted_var;
        let output = self.transform(normalized);

        self.update_running_stat(&self.running_mean, &mean);
        self.update_running_stat(&self.running_var, &biased_var);

        output
    }

    fn biases(&self) -> Tensor<Ix1> {
        self.bias.clone()
    }

    fn weights(&self) -> Tensor<Ix1> {
        self.weight.clone()
    }

    fn is_trainable(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn get_running_stats<D: Dimension + RemoveAxis>(features: usize, input: &Tensor<D>) -> Vec<f64>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        let batch_norm = BatchNorm::new("bn", features);
        batch_norm.forward(input);

        let running_mean = batch_norm.running_mean.take().into_raw_vec();
        let running_var = batch_norm.running_var.take().into_raw_vec();

        let mut outputs = vec![];
        outputs.extend(running_mean);
        outputs.extend(running_var);

        outputs.into_iter().map(|v| v.value()).collect::<Vec<f64>>()
    }

    #[test]
    fn valid_1d_batch_norm_running_statistic() {
        let mini_batch = tensor![[[1., 2., 3.], [4., 5., 6.]]];

        let outputs = get_running_stats(2, &mini_batch);
        let actuals = vec![0.2, 0.5, 1.0, 1.0];

        for (output, actual) in outputs.into_iter().zip(actuals) {
            assert_abs_diff_eq!(output, actual, epsilon = 1e-2);
        }
    }

    #[test]
    fn valid_2d_batch_norm_running_statistic() {
        let mini_batch = Tensor::<Ix4>::from_shape_vec(
            [1, 2, 2, 3],
            values![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        )
        .unwrap();

        let outputs = get_running_stats(2, &mini_batch);
        let actuals = [0.35, 0.95, 1.25, 1.25];

        for (output, actual) in outputs.into_iter().zip(actuals) {
            assert_abs_diff_eq!(output, actual, epsilon = 1e-2);
        }
    }

    #[test]
    fn valid_3d_batch_norm_running_statistic() {
        let mini_batch =
            Tensor::<Ix5>::from_shape_vec([1, 1, 2, 2, 2], values![1., 2., 3., 4., 5., 6., 7., 8.])
                .unwrap();

        let outputs = get_running_stats(1, &mini_batch);
        let actuals = [0.45, 1.5];

        for (output, actual) in outputs.into_iter().zip(actuals) {
            assert_abs_diff_eq!(output, actual, epsilon = 1e-2);
        }
    }
}
