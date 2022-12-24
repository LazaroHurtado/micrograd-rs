use crate::value::Value;
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};

pub enum WeightInit {
    GlorotUniform,
    GlorotNormal,
}

impl WeightInit {
    pub fn sample(&self, fanning: [usize; 2]) -> Value {
        let [fan_in, fan_out] = fanning;

        match self {
            Self::GlorotUniform => self.glorot_uniform(fan_in, fan_out),
            Self::GlorotNormal => self.glorot_normal(fan_in, fan_out),
        }
    }

    fn glorot_uniform(&self, fan_in: usize, fan_out: usize) -> Value {
        let limit = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
        let uniform = Uniform::new(-limit, limit);

        let mut rng = rand::thread_rng();

        let weight = uniform.sample(&mut rng);
        Value::from(weight)
    }

    fn glorot_normal(&self, fan_in: usize, fan_out: usize) -> Value {
        let stdev = (2.0 / (fan_in as f64 + fan_out as f64)).sqrt();
        let normal = Normal::new(0.0, stdev).unwrap();

        let mut rng = rand::thread_rng();

        let weight = normal.sample(&mut rng);
        Value::from(weight)
    }
}
