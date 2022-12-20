use super::mlp::Module;
use super::operation::Op;
use super::value::Value;
use rand::distributions::{Distribution, Uniform};
use std::{f64::consts::E, fmt, iter::zip};

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    ReLu,
    TanH,
}

impl Activation {
    pub fn activate(&self, value: Value) -> Value {
        let data = value.0.borrow();

        match self {
            Self::ReLu => {
                let activated = data.value.max(0.0);
                Value::with_op(activated, Op::ReLu(value.clone()))
            }
            Self::TanH => {
                let activated = (E.powf(2.0 * data.value) - 1.0) / (E.powf(2.0 * data.value) + 1.0);
                Value::with_op(activated, Op::TanH(value.clone()))
            }
        }
    }
}

impl Module for Activation {
    fn forward(&self, outputs: Vec<Value>) -> Vec<Value> {
        outputs
            .into_iter()
            .map(|neuron_output| self.activate(neuron_output))
            .collect()
    }
}

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let uniform = Uniform::from(-1.0..=1.0);
        let mut rng = rand::thread_rng();

        let weights: Vec<Value> = (0..nin)
            .map(|_| {
                let sample = uniform.sample(&mut rng);
                Value::new(sample)
            })
            .collect();

        let sample = uniform.sample(&mut rng);
        let bias = Value::new(sample);

        Neuron { weights, bias }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.bias.clone()];
        params.append(&mut self.weights.clone());

        params
    }

    pub fn call(&self, inputs: Vec<Value>) -> Value {
        let sum: Value = zip(inputs.clone(), self.weights.clone())
            .map(|(x, w)| x * w)
            .sum();

        let output = &sum + &self.bias;
        output
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, weight) in self.weights.iter().enumerate() {
            writeln!(f, "\t\t[Weight-{:?}]   {:?}", idx, weight)?;
        }
        Ok(())
    }
}
