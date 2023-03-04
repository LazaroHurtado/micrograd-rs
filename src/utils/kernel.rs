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

        for filter in self.filter.receptive_field(input) {
            let convolution = &filter * &self.weights;
            output.push(convolution.sum() + self.bias.clone());
        }

        let channelless_input = input.raw_dim().try_remove_axis(Axis(0));
        let channel_shape = self.filter.output_shape(channelless_input);

        Tensor::from_shape_vec(channel_shape, output).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::{Conv1D, Conv2D, Conv3D, Layer};

    #[test]
    fn valid_kernel_weight_tensor_size() {
        let in_channels = 1;
        let (n, m) = (2, 3);

        let filter = Filter {
            size: Dim((n, m)),
            stride: Dim((1, 0)),
            ..Default::default()
        };
        let kernel_2x2: Kernel<Ix2, Ix3> = Kernel::new(in_channels, 4, filter);

        assert_eq!(kernel_2x2.weights.shape(), &[in_channels, n, m]);
    }

    #[test]
    fn valid_convolution() {
        let mut kernel: Kernel<Ix2, Ix3> = Kernel::new(1, 1, Filter::new((2, 2), (1, 1), (1, 1)));
        kernel.weights = tensor![[[0.3954, -0.1740], [-0.1890, 0.4909]]];
        kernel.bias = val!(-0.1188);

        let input = tensor![[
            [-1.5237, 0.9591, -2.0597, 0.8249],
            [-0.4506, -0.6975, 1.0153, -0.2838],
            [-0.5344, -0.5019, -0.4378, 0.3062],
            [0.0597, 1.4820, 0.4158, 1.4295],
            [0.0612, -0.4898, -0.2115, -0.4827]
        ]];

        let outputs = kernel.convolve(&input).mapv(|v| v.value()).into_raw_vec();
        let actuals = vec![
            -1.14539373,
            1.24905421,
            -1.40794710,
            -0.32098335,
            -0.69131062,
            0.56508859,
            0.47345934,
            -0.31705584,
            0.27797043,
            -0.60507223,
            0.38358044,
            -0.40010961,
        ];

        for (output, actual) in outputs.into_iter().zip(actuals) {
            assert_abs_diff_eq!(output, actual, epsilon = 1e-6);
        }
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

        let filter = Filter {
            size: Dim(n),
            stride: Dim(1),
            ..Default::default()
        };
        let kernel = Kernel {
            filter,
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

        let filter = Filter {
            size: Dim((n, m)),
            stride: Dim((1, 1)),
            ..Default::default()
        };
        let kernel = Kernel {
            filter,
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

        let filter = Filter {
            size: Dim((n, m, l)),
            stride: Dim((1, 1, 1)),
            ..Default::default()
        };
        let kernel = Kernel {
            filter,
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
