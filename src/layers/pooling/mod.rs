mod avg_pool;
mod max_pool;

use ndarray::{Dimension, RemoveAxis};

use crate::Tensor;

pub use self::avg_pool::AvgPool;
pub use self::max_pool::MaxPool;
pub use super::Layer;

pub trait PoolingFn<D: Dimension> {
    fn pool_name(&self) -> String;
    fn pool(&self, input: Tensor<D>) -> Tensor<D>;
    fn output_shape<E: Dimension>(&self, input_dim: &E) -> E;
}

macro_rules! impl_layer_for_pool {
    [$($pooling: ident),+] => {
        $(impl<SingleChannelDim, MultiChannelDim, BatchedDim> Layer<BatchedDim, BatchedDim> for $pooling<SingleChannelDim>
where
    BatchedDim: Dimension<Smaller = MultiChannelDim> + RemoveAxis,
    MultiChannelDim: Dimension<Larger = BatchedDim, Smaller = SingleChannelDim> + RemoveAxis,
    SingleChannelDim: Dimension<Larger = MultiChannelDim>
{
    fn forward(&self, input: &Tensor<BatchedDim>) -> Tensor<BatchedDim> {
        let mut output_channels = vec![];

        for single_batch in input.outer_iter() {
            for channel in single_batch.outer_iter() {
                let pooled_channel = self.pool(channel.into_owned());
                output_channels.append(&mut pooled_channel.into_raw_vec());
            }
        }

        let output_dim = self.output_shape(&input.raw_dim());
        Tensor::from_shape_vec(output_dim, output_channels).unwrap()
    }

    fn name(&self) -> String {
        self.pool_name()
    }
})*
    };
}

impl_layer_for_pool![AvgPool, MaxPool];
