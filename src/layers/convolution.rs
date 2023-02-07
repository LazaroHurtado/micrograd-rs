use super::{Filter, Kernel, Layer};
use crate::prelude::*;
use ndarray::{concatenate, IntoDimension, RemoveAxis};

pub type Conv1D = Convolution<Ix1, Ix2>;
pub type Conv2D = Convolution<Ix2, Ix3>;
pub type Conv3D = Convolution<Ix3, Ix4>;

pub struct Convolution<D, E> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub padding: D,
    pub kernels: Vec<Kernel<D, E>>,
}

impl<D, E> Convolution<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D> + RemoveAxis,
{
    pub fn new<J: IntoDimension<Dim = D> + Clone>(
        in_channels: usize,
        out_channels: usize,
        padding: J,
        filter: Filter<D>,
    ) -> Self {
        let kernels = (0..out_channels)
            .map(|_| Kernel::new(in_channels, out_channels, filter.clone()))
            .collect::<Vec<Kernel<D, E>>>();

        Convolution {
            in_channels,
            out_channels,
            padding: padding.into_dimension(),
            kernels,
        }
    }

    pub fn pad_input(&self, input: Tensor<E>) -> Tensor<E> {
        let mut padded_input = input;

        for (axis, &padding) in self.padding.slice().iter().enumerate() {
            let mut padding_dim = padded_input.raw_dim();
            padding_dim[axis + 1] = padding;

            let total_padding = padding_dim.slice().iter().product();

            let padding_tensor: Tensor<E> =
                Tensor::from_shape_vec(padding_dim, vec![val!(0.0); total_padding]).unwrap();
            padded_input = concatenate![Axis(axis + 1), padding_tensor, padded_input];
            padded_input = concatenate![Axis(axis + 1), padded_input, padding_tensor];
        }

        padded_input
    }
}

impl<D, E> Layer<E, E> for Convolution<D, E>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D> + RemoveAxis,
{
    fn parameters(&self) -> Tensor<Ix1> {
        self.kernels
            .iter()
            .flat_map(|kernel| kernel.parameters().to_vec())
            .collect()
    }

    fn forward(&self, input: &Tensor<E>) -> Tensor<E> {
        let padded_input = self.pad_input(input.clone());

        let mut single_channel_dim = <D>::zeros(D::NDIM.unwrap());
        let mut output_channels = vec![];

        self.kernels.iter().for_each(|kernel| {
            let channel = kernel.convolve(&padded_input);

            single_channel_dim = channel.raw_dim();
            output_channels.append(&mut channel.into_raw_vec());
        });

        let mut output_dim = single_channel_dim.insert_axis(Axis(0));
        output_dim[0] = self.out_channels;
        Tensor::from_shape_vec(output_dim, output_channels).unwrap()
    }
}
