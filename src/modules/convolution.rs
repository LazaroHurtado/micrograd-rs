use super::Module;
use crate::prelude::*;
use crate::utils::WeightInit;
use ndarray::IntoDimension;

pub type Conv1D = Convolution<Ix1, Ix2>;
pub type Conv2D = Convolution<Ix2, Ix3>;
pub type Conv3D = Convolution<Ix3, Ix4>;

//TODO: Implement support for user defined padding and dilation
struct Kernel<D, E> {
    size: D,
    stride: D,
    padding: D,
    dilation: D,
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
        let mut size_with_channel = size_d.clone().insert_axis(Axis(0));
        size_with_channel[0] = in_channels;

        let weights = Tensor::from_shape_simple_fn(size_with_channel.clone(), || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });
        let bias = Value::zero();

        let default_padding = <D>::zeros(D::NDIM.unwrap());
        let mut default_dilation = <D>::zeros(D::NDIM.unwrap());
        for ix in 0..default_dilation.ndim() {
            default_dilation[ix] = 1;
        }

        Kernel {
            size: size_d,
            stride: stride.into_dimension(),
            padding: default_padding,
            dilation: default_dilation,
            weights,
            bias,
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone().into_raw_vec();
        params.push(self.bias.clone());

        params
    }

    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    fn input_to_output_dim_conversion(
        input_dim: usize,
        padding: usize,
        dilation: usize,
        size: usize,
        stride: usize,
    ) -> usize {
        let padding_expansion = 2 * padding;
        let total_kernel_size = dilation * (size - 1);

        ((input_dim + padding_expansion - total_kernel_size - 1) / stride) + 1
    }

    pub fn output_shape(&self, input_dim: D) -> D {
        let mut output_dim = input_dim.clone();

        let input = input_dim.slice();
        let size = self.size.slice();
        let padding = self.padding.slice();
        let dilation = self.dilation.slice();
        let stride = self.stride.slice();

        for ix in 0..output_dim.ndim() {
            output_dim[ix] = Self::input_to_output_dim_conversion(
                input[ix],
                padding[ix],
                dilation[ix],
                size[ix],
                stride[ix],
            );
        }

        output_dim
    }

    pub fn convolve(&self, input: Tensor<E>) -> Tensor<D> {
        let mut output = vec![];

        let window_size = self.weights.dim();
        let stride = self.stride.insert_axis(Axis(0));

        for window in input.windows_with_stride(window_size, stride.into_pattern()) {
            let convolution = (&window * &self.weights) + self.bias.clone();
            output.push(convolution.sum());
        }

        let channelless_input = input.raw_dim().try_remove_axis(Axis(0));
        let channel_shape = self.output_shape(channelless_input);
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

    pub fn output_shape(&self, input_dim: E) -> E {
        let channelless_input = input_dim.try_remove_axis(Axis(0));
        let single_channel_shape = match self.kernels.first() {
            Some(kernel) => kernel.output_shape(channelless_input),
            None => <D>::default(),
        };

        let mut output_dim = single_channel_shape.insert_axis(Axis(0));
        output_dim[0] = self.out_channels;

        output_dim
    }
}

impl<D, E> Module for Convolution<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D>,
{
    type Dim = E;

    fn parameters(&self) -> Vec<Value> {
        let mut params = vec![];

        self.kernels.iter().for_each(|kernel| {
            params.append(&mut kernel.parameters());
        });
        params
    }

    fn forward(&self, input: Tensor<Self::Dim>) -> Tensor<Self::Dim> {
        let mut out_channels = vec![];

        self.kernels.iter().for_each(|kernel| {
            let channel = kernel.convolve(input.clone());
            out_channels.append(&mut channel.into_raw_vec());
        });

        Tensor::from_shape_vec(self.output_shape(input.raw_dim()), out_channels).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn valid_kernel_weight_tensor_size() {
        let in_channels = 1;
        let (n, m) = (2, 3);

        let kernel_2x2: Kernel<Ix2, Ix3> = Kernel::new(in_channels, 4, (n, m), (1, 0));

        assert_eq!(kernel_2x2.weights.shape(), &[in_channels, n, m]);
    }

    #[test]
    fn valid_kernel_output_size_for_input() {
        let in_channels = 3;
        let (input_h, input_w) = (7, 13);
        let input = Tensor::zeros((in_channels, input_h, input_w));

        let channelless_input = input.raw_dim().try_remove_axis(Axis(0));

        let (n, m) = (2, 3);
        let (padding, dilation, kernel_size, stride) = (0, 1, (n, m), (1, 1));

        let kernel_2x2: Kernel<Ix2, Ix3> = Kernel::new(in_channels, 4, (n, m), stride);

        let output_h =
            ((input_h + 2 * padding - dilation * (kernel_size.0 - 1) - 1) / stride.0) + 1;
        let output_w =
            ((input_w + 2 * padding - dilation * (kernel_size.1 - 1) - 1) / stride.1) + 1;

        assert_eq!(
            kernel_2x2.output_shape(channelless_input).slice(),
            [output_h, output_w]
        );
    }

    #[test]
    fn valid_output_size_for_input() {
        let (in_channels, out_channels) = (3, 4);
        let (input_h, input_w) = (7, 13);
        let input = Tensor::zeros((in_channels, input_h, input_w));

        let (n, m) = (2, 3);
        let (padding, dilation, kernel_size, stride) = (0, 1, (n, m), (1, 1));

        let conv2d = Conv2D::new(in_channels, out_channels, (n, m), stride);

        let output_h =
            ((input_h + 2 * padding - dilation * (kernel_size.0 - 1) - 1) / stride.0) + 1;
        let output_w =
            ((input_w + 2 * padding - dilation * (kernel_size.1 - 1) - 1) / stride.1) + 1;

        assert_eq!(
            conv2d.output_shape(input.raw_dim()).slice(),
            [out_channels, output_h, output_w]
        );
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

    #[test]
    fn returns_all_parameters_in_a_single_kernel() {
        let (in_channels, out_channels) = (3, 13);

        let (n, m) = (3, 8);
        let kernel: Kernel<Ix2, Ix3> = Kernel::new(in_channels, out_channels, (n, m), (1, 0));

        let parameters_in_weight_tensor = in_channels * n * m;
        let total_parameters = parameters_in_weight_tensor + 1; // one bias parameter in a kernel

        assert_eq!(kernel.parameters().len(), total_parameters);
    }

    #[test]
    fn returns_all_parameters_in_each_kernel() {
        let (in_channels, out_channels) = (3, 13);

        let (n, m, k) = (3, 8, 4);
        let conv3d = Conv3D::new(in_channels, out_channels, (n, m, k), (1, 0, 0));

        let parameters_per_kernel = in_channels * n * m * k;
        let total_kernel_parameters = out_channels * parameters_per_kernel;
        let total_parameters = total_kernel_parameters + out_channels; // one bias parameter per
                                                                       // kernel

        assert_eq!(conv3d.parameters().len(), total_parameters);
    }
}
