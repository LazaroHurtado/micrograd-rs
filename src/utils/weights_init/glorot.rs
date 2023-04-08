use crate::{val, Value};
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};

use super::{Fanning, WeightInit};

pub struct GlorotNormal;
pub struct GlorotUniform;

impl WeightInit for GlorotNormal {
    fn sample<F: Into<Fanning>>(&self, fanning: F) -> crate::Value {
        let Fanning(fan_in, fan_out) = fanning.into();
        let stdev = (2.0 / (fan_in as f64 + fan_out as f64)).sqrt();
        let normal = Normal::new(0.0, stdev).unwrap();

        let mut rng = rand::thread_rng();

        let weight = normal.sample(&mut rng);
        val!(weight)
    }
}

impl WeightInit for GlorotUniform {
    fn sample<F: Into<Fanning>>(&self, fanning: F) -> crate::Value {
        let Fanning(fan_in, fan_out) = fanning.into();
        let limit = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
        let uniform = Uniform::new(-limit, limit);

        let mut rng = rand::thread_rng();

        let weight = uniform.sample(&mut rng);
        val!(weight)
    }
}
