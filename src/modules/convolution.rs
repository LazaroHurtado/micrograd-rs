use super::Module;
use crate::prelude::*;
use crate::utils::WeightInit;
use ndarray::IntoDimension;

pub type Conv1D = Convolution<Ix1, Ix2>;
pub type Conv2D = Convolution<Ix2, Ix3>;
pub type Conv3D = Convolution<Ix3, Ix4>;

//TODO: Implement support for user defined padding and dilation
pub struct Filter<D> {
    pub size: D,
    pub stride: D,
    pub padding: D,
    pub dilation: D,
}

impl<D> Filter<D>
where
    D: Dimension,
{
    pub fn new<E: IntoDimension<Dim = D>>(size: E, stride: E) -> Self {
        let default_padding = <D>::zeros(D::NDIM.unwrap());
        let mut default_dilation = <D>::zeros(D::NDIM.unwrap());
        for ix in 0..default_dilation.ndim() {
            default_dilation[ix] = 1;
        }

        Filter {
            size: size.into_dimension(),
            stride: stride.into_dimension(),
            padding: default_padding,
            dilation: default_dilation,
        }
    }

    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    pub fn output_shape(&self, input_dim: D) -> D {
        let mut output_dim = input_dim.clone();
        let n = output_dim.ndim();

        let input = input_dim.slice();
        let size = self.size.slice();
        let padding = self.padding.slice();
        let dilation = self.dilation.slice();
        let stride = self.stride.slice();

        for ix in 0..n {
            let padding_expansion = 2 * padding[ix];
            let total_kernel_size = dilation[ix] * (size[ix] - 1);

            let axis = ((input[ix] + padding_expansion - total_kernel_size - 1) / stride[ix]) + 1;
            output_dim[ix] = axis;
        }

        output_dim
    }
}

pub struct Kernel<D, E> {
    filter: Filter<D>,
    weights: Tensor<E>,
    bias: Value,
}

impl<D, E> Kernel<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D>,
{
    pub fn new<J: IntoDimension<Dim = D>>(
        in_channels: usize,
        out_channels: usize,
        size: J,
        stride: J,
    ) -> Self {
        let size_d = size.into_dimension();
        let mut size_with_channel = size_d.insert_axis(Axis(0));
        size_with_channel[0] = in_channels;

        let weights = Tensor::from_shape_simple_fn(size_with_channel.clone(), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });
        let bias = Value::zero();

        let filter = Filter::new(size_d, stride.into_dimension());

        Kernel {
            filter,
            weights,
            bias,
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone().into_raw_vec();
        params.push(self.bias.clone());

        params
    }

    pub fn convolve(&self, input: Tensor<E>) -> Tensor<D> {
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

pub struct Convolution<D, E> {
    in_channels: usize,
    out_channels: usize,
    kernels: Vec<Kernel<D, E>>,
}

impl<D, E> Convolution<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D>,
{
    pub fn new<J: IntoDimension<Dim = D> + Clone>(
        in_channels: usize,
        out_channels: usize,
        kernel_size: J,
        stride: J,
    ) -> Self {
        let kernels = (0..out_channels)
            .map(|_| {
                Kernel::new(
                    in_channels,
                    out_channels,
                    kernel_size.clone(),
                    stride.clone(),
                )
            })
            .collect::<Vec<Kernel<D, E>>>();

        Convolution {
            in_channels,
            out_channels,
            kernels,
        }
    }
}

impl<D, E> Module<E> for Convolution<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D>,
{
    fn parameters(&self) -> Vec<Value> {
        let mut params = vec![];

        self.kernels.iter().for_each(|kernel| {
            params.append(&mut kernel.parameters());
        });
        params
    }

    fn forward(&self, input: Tensor<E>) -> Tensor<E> {
        let mut single_channel_dim = <D>::zeros(D::NDIM.unwrap());
        let mut output_channels = vec![];

        self.kernels.iter().for_each(|kernel| {
            let channel = kernel.convolve(input.clone());
            single_channel_dim = channel.raw_dim();
            output_channels.append(&mut channel.into_raw_vec());
        });

        let mut output_dim = single_channel_dim.insert_axis(Axis(0));
        output_dim[0] = self.out_channels;
        Tensor::from_shape_vec(output_dim, output_channels).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_kernel_weight_tensor_size() {
        let in_channels = 1;
        let (n, m) = (2, 3);

        let kernel_2x2: Kernel<Ix2, Ix3> = Kernel::new(in_channels, 4, (n, m), (1, 0));

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

        let mut conv1d = Conv1D::new(in_channels, out_channels, n, 1);
        conv1d.kernels[0].weights = weights;

        assert_eq!(conv1d.forward(input.clone()), input);
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

        let mut conv2d = Conv2D::new(in_channels, out_channels, (n, m), (1, 1));
        conv2d.kernels[0].weights = weights;

        assert_eq!(conv2d.forward(input.clone()), input);
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

        let mut conv3d = Conv3D::new(in_channels, out_channels, (n, m, l), (1, 1, 1));
        conv3d.kernels[0].weights = weights;

        assert_eq!(conv3d.forward(input.clone()), input);
    }
}
