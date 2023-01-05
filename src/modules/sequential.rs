use super::Module;
use crate::prelude::*;
use ndarray::RemoveAxis;

#[macro_export]
macro_rules! sequential {
    ($d: tt, [$($module: expr),*]) => {{
        let mut modules: Vec<Box<dyn Module<$d>>> = vec![];
        $(
            modules.push(Box::new($module));
        )*
        Sequential::new(modules)
    }};
}

pub struct Sequential<D> {
    layers: Vec<Box<dyn Module<D>>>,
}

impl<D, E> Sequential<D>
where
    E: Dimension<Smaller = D> + RemoveAxis,
    D: Dimension<Larger = E>,
{
    pub fn new(layers: Vec<Box<dyn Module<D>>>) -> Self {
        Sequential { layers }
    }

    pub fn parameters(&self) -> Tensor<Ix1> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters().to_vec())
            .collect()
    }

    pub fn zero_grad(&self) {
        self.parameters().iter().for_each(|value| value.zero_grad());
    }

    pub fn forward(&self, inputs: Tensor<D>) -> Tensor<D> {
        self.layers
            .iter()
            .fold(inputs, |output, layer| layer.forward(output))
    }

    pub fn forward_batch(&self, batches: Tensor<E>) -> Tensor<E> {
        let mut outputs = vec![];
        let mut output_size = <D>::default();

        for batch in batches.axis_iter(Axis(0)) {
            let batch_output = self.forward(batch.to_owned());
            output_size = batch_output.raw_dim();
            outputs.append(&mut batch_output.into_raw_vec());
        }

        let batches_d = batches.raw_dim();
        let mut output_shape = output_size.insert_axis(Axis(0));
        output_shape[0] = batches_d[0];

        Tensor::from_shape_vec(output_shape, outputs).unwrap()
    }
}
