use super::{Filter, Module};
use crate::prelude::*;
use ndarray::RemoveAxis;

pub enum Pooling<D> {
    Max(Filter<D>),
    Average(Filter<D>),
}

impl<D> Pooling<D>
where
    D: Dimension,
{
    fn pool(&self, input: Tensor<D>) -> Tensor<D> {
        let filter = self.filter();
        let (size, stride) = (filter.size.clone(), filter.stride.clone());
        let mut pooled_input = vec![];

        for window in input.windows_with_stride(size, stride) {
            let pooled_window = self.call(window.into_owned());
            pooled_input.push(pooled_window);
        }

        let output_dim = filter.output_shape(input.raw_dim());
        Tensor::from_shape_vec(output_dim, pooled_input).unwrap()
    }

    fn filter(&self) -> &Filter<D> {
        match self {
            Self::Max(filter) | Self::Average(filter) => filter,
        }
    }

    fn call(&self, window: Tensor<D>) -> Value {
        match self {
            Self::Max(_) => self.max_pooling(window),
            Self::Average(_) => self.avg_pooling(window),
        }
    }

    fn max_pooling(&self, window: Tensor<D>) -> Value {
        window
            .into_raw_vec()
            .into_iter()
            .reduce(Value::max)
            .unwrap()
    }

    fn avg_pooling(&self, window: Tensor<D>) -> Value {
        let data = window.into_raw_vec();
        let n = data.len() as f64;

        data.into_iter().sum::<Value>() / n
    }
}

impl<D, E> Module<E> for Pooling<D>
where
    D: Dimension<Larger = E>,
    E: Dimension<Smaller = D> + RemoveAxis,
{
    fn forward(&self, input: Tensor<E>) -> Tensor<E> {
        let mut single_channel_dim = <D>::zeros(D::NDIM.unwrap());
        let mut output_channels = vec![];

        for channel in input.axis_iter(Axis(0)) {
            let pooled_channel = self.pool(channel.into_owned());
            single_channel_dim = pooled_channel.raw_dim();
            output_channels.append(&mut pooled_channel.into_raw_vec());
        }

        let mut output_dim = single_channel_dim.insert_axis(Axis(0));
        output_dim[0] = input.raw_dim()[0];
        Tensor::from_shape_vec(output_dim, output_channels).unwrap()
    }
}
