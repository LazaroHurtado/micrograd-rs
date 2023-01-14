use crate::value::Value;
use rand::distributions::Uniform;
use rand_distr::{Distribution, Normal};

pub enum WeightInit {
    GlorotUniform,
    GlorotNormal,
    HeUniform(FanMode),
    HeNormal(FanMode),
}

pub enum FanMode {
    In,
    Out,
}

// TODO: More elegant way of choosing fan mode for He W.I.?
impl FanMode {
    fn choose_fan(&self, fan_in: usize, fan_out: usize) -> usize {
        match self {
            FanMode::In => fan_in,
            FanMode::Out => fan_out,
        }
    }
}

impl WeightInit {
    pub fn sample(&self, fanning: [usize; 2]) -> Value {
        let [fan_in, fan_out] = fanning;

        match self {
            Self::GlorotUniform => self.glorot_uniform(fan_in, fan_out),
            Self::GlorotNormal => self.glorot_normal(fan_in, fan_out),
            Self::HeUniform(mode) => self.he_uniform(mode.choose_fan(fan_in, fan_out)),
            Self::HeNormal(mode) => self.he_normal(mode.choose_fan(fan_in, fan_out)),
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
