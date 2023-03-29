use crate::optimizers::Optimizer;
use crate::value::Value;

mod constant_lr;
mod cosine_annealing_lr;
mod exponential_lr;
mod lambda_lr;
mod linear_lr;
mod multiplicative_lr;
mod multistep_lr;
mod polynomial_lr;
mod step_lr;

pub use self::constant_lr::ConstantLR;
pub use self::cosine_annealing_lr::CosineAnnealingLR;
pub use self::exponential_lr::ExponentialLR;
pub use self::lambda_lr::LambdaLR;
pub use self::linear_lr::LinearLR;
pub use self::multiplicative_lr::MultiplicativeLR;
pub use self::multistep_lr::MultiStepLR;
pub use self::polynomial_lr::PolynomialLR;
pub use self::step_lr::StepLR;

pub trait Schedule {
    const HAS_CLOSED_FORM: bool = true;

    fn get_lr(&self, base_lr: f64, last_epoch: usize) -> f64;
    fn get_closed_form_lr(&self, _base_lr: f64, _last_epoch: usize) -> f64 {
        0.0
    }
}

pub struct LRScheduler<S: Schedule> {
    lr: Value,
    schedule: S,
    base_lr: f64,
    last_epoch: usize,
}

impl<S: Schedule> LRScheduler<S> {
    pub fn new<O: Optimizer>(optimizer: &O, schedule: S) -> Self {
        let lr = optimizer.lr();
        let base_lr = lr.value();
        let last_epoch = 0;

        let mut scheduler = LRScheduler {
            lr,
            schedule,
            base_lr,
            last_epoch,
        };

        scheduler.step();
        scheduler
    }

    pub fn step(&mut self) {
        let new_lr = self.schedule.get_lr(self.lr.value(), self.last_epoch);

        *self.lr.value_mut() = new_lr.into();
        self.last_epoch += 1;
    }

    pub fn step_with(&mut self, epoch: usize) {
        let new_lr = if S::HAS_CLOSED_FORM {
            self.schedule.get_closed_form_lr(self.base_lr, epoch)
        } else {
            self.schedule.get_lr(self.lr.value(), epoch)
        };

        *self.lr.value_mut() = new_lr.into();
        self.last_epoch = epoch;
    }

    pub fn lr(&self) -> f64 {
        self.lr.value()
    }
}
