use super::{Layer, Model};
use crate::prelude::*;
use indexmap::IndexMap;
use ndarray::RemoveAxis;
use serde_pickle::{de, DeOptions};
use std::fs::File;

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

    pub fn forward(&self, inputs: &Tensor<D>) -> Tensor<D> {
        self.layers
            .iter()
            .fold(inputs.clone(), |output, layer| layer.forward(&output))
    }

    pub fn forward_batch(&self, batches: &Tensor<E>) -> Tensor<E> {
        let mut outputs = vec![];
        let mut output_size = <D>::default();

        for batch in batches.axis_iter(Axis(0)) {
            let batch_output = self.forward(&batch.to_owned());
            output_size = batch_output.raw_dim();
            outputs.append(&mut batch_output.into_raw_vec());
        }

        let batches_d = batches.raw_dim();
        let mut output_shape = output_size.insert_axis(Axis(0));
        output_shape[0] = batches_d[0];

        Tensor::from_shape_vec(output_shape, outputs).unwrap()
    }
}

impl<D: Dimension> Model for Sequential<D> {
    fn state_dict(&self) -> IndexMap<String, Vec<f64>> {
        let mut state_dict: IndexMap<String, Vec<f64>> = IndexMap::new();

        let trainable_layers = self.layers.iter().filter(|x| x.is_trainable());

        for layer in trainable_layers {
            let (weight_key, bias_key) = (layer.name() + ".weight", layer.name() + ".bias");

            let layer_weights = layer.weights().iter().map(|v| v.value()).collect();
            state_dict.insert(weight_key, layer_weights);

            let layer_biases = layer.biases().iter().map(|v| v.value()).collect();
            state_dict.insert(bias_key, layer_biases);
        }

        state_dict
    }

    fn load_state_dict(&mut self, path: &str) {
        let file = File::open(path).unwrap();
        let state_dict: IndexMap<String, Vec<f64>> =
            de::from_reader(file, DeOptions::new()).unwrap();

        let trainable_layers = self.layers.iter().filter(|x| x.is_trainable());

        for layer in trainable_layers {
            let (weight_key, bias_key) = (layer.name() + ".weight", layer.name() + ".bias");

            if state_dict.get(&weight_key).unwrap().len() != layer.weights().len() {
                panic!("Wrong loaded weight count for layer \"{}\".", layer.name());
            }
            if state_dict.get(&bias_key).unwrap().len() != layer.biases().len() {
                panic!("Wrong loaded bias count for layer \"{}\".", layer.name());
            }

            layer.set_weights(state_dict.get(&weight_key).unwrap());
            layer.set_biases(state_dict.get(&bias_key).unwrap());
        }
    }
}
