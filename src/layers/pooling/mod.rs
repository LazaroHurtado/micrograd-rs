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
        $(impl<C, D> Layer<C, C> for $pooling<D>
where
    C: Dimension<Smaller = D> + RemoveAxis,
    D: Dimension<Larger = C>
{
    fn forward(&self, input: &Tensor<C>) -> Tensor<C> {
        let mut output_channels = vec![];

            for channel in input.outer_iter() {
                let pooled_channel = self.pool(channel.into_owned());
                output_channels.append(&mut pooled_channel.into_raw_vec());
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
