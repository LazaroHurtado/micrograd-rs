use crate::value::Value;
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};

pub enum WeightInit {
    GlorotUniform,
    GlorotNormal,
    HeUniform,
    HeNormal,
}

pub struct Fanning(usize, usize);
impl Into<Fanning> for [usize;2] {
    fn into(self) -> Fanning {
        Fanning(self[0], self[1])
    }
}
impl Into<Fanning> for [usize;1] {
    fn into(self) -> Fanning {
        Fanning(self[0], self[0])
    }
}

impl WeightInit {
    pub fn sample<F: Into<Fanning>>(&self, fanning: F) -> Value {
        let Fanning(fan_in, fan_out) = fanning.into();

        match self {
            Self::GlorotUniform => self.glorot_uniform(fan_in, fan_out),
            Self::GlorotNormal => self.glorot_normal(fan_in, fan_out),
            Self::HeUniform => self.he_uniform(fan_in),
            Self::HeNormal => self.he_normal(fan_in),
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

    fn he_uniform(&self, fan_num: usize) -> Value {
        let limit = (3.0 / (fan_num as f64)).sqrt();
        let uniform = Uniform::new(-limit, limit);

        let mut rng = rand::thread_rng();

        let weight = uniform.sample(&mut rng);
        Value::from(weight)
    }

    fn he_normal(&self, fan_num: usize) -> Value {
        let stdev = 1.0 / (fan_num as f64).sqrt();
        let normal = Normal::new(0.0, stdev).unwrap();

        let mut rng = rand::thread_rng();

        let weight = normal.sample(&mut rng);
        Value::from(weight)
    }
}
