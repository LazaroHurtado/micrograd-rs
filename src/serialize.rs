use serde_pickle::{from_reader, to_writer};
use std::collections::BTreeMap;
use std::fs::File;

pub fn save(state_dict: BTreeMap<String, Vec<f64>>, path: &str) {
    let mut file = File::create(path).unwrap();
    to_writer(&mut file, &state_dict, true).unwrap();
}

pub fn load(path: &str) -> BTreeMap<String, Vec<f64>> {
    let file = File::open(path).unwrap();
    let deserialized_dict: BTreeMap<String, Vec<f64>> = from_reader(file).unwrap();
    deserialized_dict
}
