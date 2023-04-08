use super::Layer;
use crate::{prelude::*, utils::WeightInit};
use ndarray::{concatenate, IntoDimension, RemoveAxis, Slice};

pub type Conv1D = Convolution<Ix1, Ix3>;
pub type Conv2D = Convolution<Ix2, Ix4>;
pub type Conv3D = Convolution<Ix3, Ix5>;

pub struct Convolution<SingleChannelDim: Dimension, BatchedDim: Dimension> {
    pub name: String,
    pub in_channels: usize,
    pub out_channels: usize,

    pub kernel_size: SingleChannelDim,
    pub padding: SingleChannelDim,
    pub stride: SingleChannelDim,
    pub dilation: SingleChannelDim,

    pub weights: Tensor<BatchedDim>,
    pub biases: Tensor<Ix1>,
}

impl<SingleChannelDim, MultiChannelDim, BatchedDim> Convolution<SingleChannelDim, BatchedDim>
where
    SingleChannelDim: Dimension<Larger = MultiChannelDim> + RemoveAxis,
    MultiChannelDim: Dimension<Smaller = SingleChannelDim, Larger = BatchedDim> + RemoveAxis,
    BatchedDim: Dimension<Smaller = MultiChannelDim> + RemoveAxis,
{
    pub fn new<J: IntoDimension<Dim = SingleChannelDim> + Clone>(
        name: impl ToString,
        in_channels: usize,
        out_channels: usize,
        kernel_size: J,
        padding: J,
        stride: J,
        dilation: J,
    ) -> Self {
        let name = name.to_string();

        let weights_dim = {
            let kernel_dim = kernel_size.clone().into_dimension();
            let mut weights_dim = kernel_dim.insert_axis(Axis(0)).insert_axis(Axis(0));

            let weights_slice = weights_dim.slice_mut();
            weights_slice[0] = out_channels;
            weights_slice[1] = in_channels;

            weights_dim
        };

        let weights = Tensor::from_shape_simple_fn(weights_dim, || {
            WeightInit::GlorotUniform.sample([in_channels, out_channels])
        });
        let biases = Tensor::from_shape_simple_fn(out_channels, Value::zero);

        Convolution {
            name,
            in_channels,
            out_channels,

            kernel_size: kernel_size.into_dimension(),
            padding: padding.into_dimension(),
            stride: stride.into_dimension(),
            dilation: dilation.into_dimension(),

            weights,
            biases,
        }
    }

    fn pad_input(&self, input: &Tensor<MultiChannelDim>) -> Tensor<MultiChannelDim> {
        let mut padded_input = input.clone();

        for (axis, &padding) in self.padding.slice().iter().enumerate() {
            let mut padding_dim = padded_input.raw_dim();
            padding_dim[axis + 1] = padding;

            let total_padding = padding_dim.slice().iter().product();

            let padding_tensor: Tensor<MultiChannelDim> =
                Tensor::from_shape_vec(padding_dim, vec![val!(0.0); total_padding]).unwrap();
            padded_input = concatenate![Axis(axis + 1), padding_tensor, padded_input];
            padded_input = concatenate![Axis(axis + 1), padded_input, padding_tensor];
        }

        padded_input
    }

    fn output_shape<D: Dimension>(&self, input_dim: &D) -> D {
        let mut output_dim = input_dim.clone();
        let n = SingleChannelDim::NDIM.unwrap();
        let offset = input_dim.ndim() - n;

        for ix in 0..n {
            let padding_expansion = 2 * self.padding[ix];
            let total_filter_size = self.dilation[ix] * (self.kernel_size[ix] - 1);

            let axis = ((input_dim[ix + offset] + padding_expansion - total_filter_size - 1)
                / self.stride[ix])
                + 1;
            output_dim[ix + offset] = axis;
        }

        output_dim
    }

    fn window_shape(&self) -> MultiChannelDim {
        let mut size = self.kernel_size.clone();

        for ix in 0..size.ndim() {
            size[ix] += (self.dilation[ix] - 1) * (size[ix] - 1);
        }

        size.insert_axis(Axis(0))
    }

    fn dilate_filter(&self, filter: &Tensor<MultiChannelDim>) -> Tensor<MultiChannelDim> {
        let dilation = self.dilation.insert_axis(Axis(0));
        let dilated_filter =
            filter.slice_each_axis(|ax| Slice::new(0, None, dilation[ax.axis.index()] as isize));

        dilated_filter.to_owned()
    }

    pub fn convolve(&self, input: &Tensor<MultiChannelDim>) -> Tensor<MultiChannelDim> {
        let input_dim = input.raw_dim();
        let output_shape = self.output_shape(&input_dim);

        let window_shape = self.window_shape();
        let stride = self.stride.insert_axis(Axis(0));

        let mut output = vec![];
        let input = self.pad_input(&input.to_owned());

        for (weight, bias) in self.weights.outer_iter().zip(self.biases.clone()) {
            for window in input.windows_with_stride(window_shape.clone(), stride.clone()) {
                let window = self.dilate_filter(&window.to_owned());

                let convolved = &window * weight.to_owned();
                output.push(convolved.sum() + bias.to_owned());
            }
        }

        Tensor::from_shape_vec(output_shape, output).unwrap()
    }
}

impl<SingleChannelDim, MultiChannelDim, BatchedDim> Layer<BatchedDim, BatchedDim>
    for Convolution<SingleChannelDim, BatchedDim>
where
    SingleChannelDim: Dimension<Larger = MultiChannelDim> + RemoveAxis,
    MultiChannelDim: Dimension<Smaller = SingleChannelDim, Larger = BatchedDim> + RemoveAxis,
    BatchedDim: Dimension<Smaller = MultiChannelDim> + RemoveAxis,
{
    fn forward(&self, input: &Tensor<BatchedDim>) -> Tensor<BatchedDim> {
        let mut convolved_batches: Option<Tensor<BatchedDim>> = None;

        for single_batch in input.outer_iter() {
            let convolved_single_batch = self.convolve(&single_batch.to_owned());
            let convolved_single_batch = convolved_single_batch.insert_axis(Axis(0));

            convolved_batches = match convolved_batches {
                None => Some(convolved_single_batch),
                Some(prev_batches) => {
                    Some(concatenate![Axis(0), prev_batches, convolved_single_batch])
                }
            }
        }

        convolved_batches.unwrap()
    }

    fn weights(&self) -> Tensor<Ix1> {
        self.weights.clone().into_shape(self.weights.len()).unwrap()
    }

    fn biases(&self) -> Tensor<Ix1> {
        self.biases.clone().into_shape(self.biases.len()).unwrap()
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
    use super::*;

    #[test]
    fn valid_conv1d_padding() {
        let name = "conv1d";
        let padding = 2;
        let (in_channels, out_channels) = (1, 1);

        let conv1d = Conv1D::new(name, in_channels, out_channels, 3, padding, 2, 1);

        let input = tensor![[1., 1., 1.]];
        let padded_input = tensor![[0., 0., 1., 1., 1., 0., 0.]];

        assert_eq!(conv1d.pad_input(&input), padded_input);
    }

    #[test]
    fn valid_conv2d_padding() {
        let name = "conv2d";
        let padding = (1, 2);
        let (in_channels, out_channels) = (1, 1);

        let conv2d = Conv2D::new(
            name,
            in_channels,
            out_channels,
            (2, 2),
            padding,
            (1, 1),
            (1, 1),
        );

        let input = tensor![[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]];
        let padded_input = tensor![
            [
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 2., 0., 0.],
                [0., 0., 3., 4., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
            ],
            [
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 5., 6., 0., 0.],
                [0., 0., 7., 8., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
            ]
        ];

        assert_eq!(conv2d.pad_input(&input), padded_input);
    }

    #[test]
    fn valid_filter_output_size_for_input() {
        let in_channels = 2;
        let input = (1, in_channels, 7, 13);

        let (n, m) = (2, 3);
        let (padding, dilation, kernel_size, stride) = ((2, 2), (1, 1), (n, m), (1, 2));

        let conv2d = Conv2D::new(
            "conv2d",
            in_channels,
            1,
            kernel_size,
            padding,
            stride,
            dilation,
        );

        let output_h =
            ((input.2 + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0) + 1;
        let output_w =
            ((input.3 + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1) + 1;

        assert_eq!(
            conv2d.output_shape(&Dim(input)).slice(),
            [1, in_channels, output_h, output_w]
        );
    }
}
