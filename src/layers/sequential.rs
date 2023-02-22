use super::Layer;
use crate::prelude::*;
use ndarray::RemoveAxis;
use std::collections::BTreeMap;

#[macro_export]
macro_rules! sequential {
    ($d: tt, [$($layer: expr),*]) => {{
        let mut layers: Vec<Box<dyn Layer<$d, $d>>> = vec![];
        $(
            layers.push(Box::new($layer));
        )*
        Sequential::new(layers)
    }};
}

pub struct Sequential<D> {
    layers: Vec<Box<dyn Layer<D, D>>>,
}

impl<D, E> Sequential<D>
where
    E: Dimension<Smaller = D> + RemoveAxis,
    D: Dimension<Larger = E>,
{
    pub fn new(layers: Vec<Box<dyn Layer<D, D>>>) -> Self {
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
            .fold(inputs, |output, layer| layer.forward(&output))
    }

    pub fn forward_batch(&self, batches: &Tensor<E>) -> Tensor<E> {
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

    pub fn state_dict(&self) -> BTreeMap<String, Vec<f64>> {
        let mut state_dict: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let layers_iter = self.layers.iter();
        let mut id = 0;
        for layer in layers_iter {
            let layer_name = layer.name();
            if !layer_name.eq("NO NAME") {
                id += 1;
                let weights_key = format!("{}{}{}", layer_name, "weights", id);
                let biases_key = format!("{}{}{}", layer_name, "biases", id);
                let shape_key = format!("{}{}{}", layer_name, "shape", id);
                state_dict.insert(weights_key, layer.weights().1);
                state_dict.insert(biases_key, layer.biases());
                state_dict.insert(shape_key, layer.weights().0);
            }
        }

        state_dict
    }
}
