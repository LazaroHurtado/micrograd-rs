use indexmap::IndexMap;
use serde_pickle::{ser, SerOptions};
use std::fs::File;

pub trait Model {
    fn save_state_dict(&self, path: &str) {
        let mut file = File::create(path).unwrap();

        let state_dict = self.state_dict();

        ser::to_writer(&mut file, &state_dict, SerOptions::new()).unwrap();
    }

    fn state_dict(&self) -> IndexMap<String, Vec<f64>>;

    fn load_state_dict(&mut self, path: &str);
}
