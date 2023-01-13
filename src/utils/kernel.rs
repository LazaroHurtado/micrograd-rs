use super::{Filter, WeightInit};
use crate::prelude::*;
use ndarray::RemoveAxis;

pub struct Kernel<D, E> {
    filter: Filter<D>,
    weights: Tensor<E>,
    bias: Value,
}

impl<D, E> Kernel<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D> + RemoveAxis,
{
    pub fn new(in_channels: usize, out_channels: usize, filter: Filter<D>) -> Self {
        let size_d = &filter.size;
        let mut size_with_channel = size_d.insert_axis(Axis(0));
        size_with_channel[0] = in_channels;

        let weights = Tensor::from_shape_simple_fn(size_with_channel.clone(), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });
        let bias = Value::zero();

        Kernel {
            filter,
            weights,
            bias,
        }
    }

    pub fn parameters(&self) -> Tensor<Ix1> {
        let mut params = self.weights.clone().into_raw_vec();
        params.push(self.bias.clone());

        Tensor::from_vec(params)
    }

    pub fn convolve(&self, input: &Tensor<E>) -> Tensor<D> {
        let mut output = vec![];

        let window_size = self.weights.dim();
        let stride = self.filter.stride.insert_axis(Axis(0));

        for window in input.windows_with_stride(window_size, stride.into_pattern()) {
            let convolution = (&window * &self.weights) + self.bias.clone();
            output.push(convolution.sum());
        }

        let channelless_input = input.raw_dim().try_remove_axis(Axis(0));
        let channel_shape = self.filter.output_shape(channelless_input);
        Tensor::from_shape_vec(channel_shape, output).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Conv1D, Conv2D, Conv3D, Module};

    #[test]
    fn valid_kernel_weight_tensor_size() {
        let in_channels = 1;
        let (n, m) = (2, 3);

        let kernel_2x2: Kernel<Ix2, Ix3> = Kernel::new(in_channels, 4, Filter::new((n, m), (1, 0)));

        assert_eq!(kernel_2x2.weights.shape(), &[in_channels, n, m]);
    }

    #[test]
    fn conv1d_with_size_one_and_weights_tensor_of_ones_should_output_input_tensor_when_one_channel()
    {
        let (in_channels, out_channels) = (1, 1);
        let input_w = 13;
        let input = Tensor::from_shape_simple_fn((in_channels, input_w), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });

        let n = 1;
        let weights = Tensor::ones((in_channels, n));
        let kernel = Kernel {
            filter: Filter::new(n, 1),
            weights,
            bias: Value::zero(),
        };

        let conv1d = Conv1D {
            in_channels,
            out_channels,
            padding: Dim(0),
            kernels: vec![kernel],
        };

        assert_eq!(conv1d.forward(&input), input);
    }

    #[test]
    fn conv2d_with_size_one_and_weights_tensor_of_ones_should_output_input_tensor_when_one_channel()
    {
        let (in_channels, out_channels) = (1, 1);
        let (input_h, input_w) = (7, 13);
        let input = Tensor::from_shape_simple_fn((in_channels, input_h, input_w), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });

        let (n, m) = (1, 1);
        let weights = Tensor::ones((in_channels, n, m));
        let kernel = Kernel {
            filter: Filter::new((m, m), (1, 1)),
            weights,
            bias: Value::zero(),
        };

        let conv2d = Conv2D {
            in_channels,
            out_channels,
            padding: Dim((0, 0)),
            kernels: vec![kernel],
        };

        assert_eq!(conv2d.forward(&input), input);
    }

    #[test]
    fn conv3d_with_size_one_and_weights_tensor_of_ones_should_output_input_tensor_when_one_channel()
    {
        let (in_channels, out_channels) = (1, 1);
        let (input_h, input_w, input_d) = (7, 13, 3);
        let input = Tensor::from_shape_simple_fn((in_channels, input_h, input_w, input_d), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });

        let (n, m, l) = (1, 1, 1);
        let weights = Tensor::ones((in_channels, n, m, l));
        let kernel = Kernel {
            filter: Filter::new((n, m, l), (1, 1, 1)),
            weights,
            bias: Value::zero(),
        };

        let conv3d = Conv3D {
            in_channels,
            out_channels,
            padding: Dim((0, 0, 0)),
            kernels: vec![kernel],
        };

        assert_eq!(conv3d.forward(&input), input);
    }
}
