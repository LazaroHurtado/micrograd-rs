use super::Schedule;

pub struct ConstantLR {
    total_iters: usize,
    factor: f64,
}

impl ConstantLR {
    pub fn new(total_iters: usize, factor: f64) -> Self {
        if !(0.0..=1.0).contains(&factor) {
            panic!("Constant factor expected to be between 0.0 and 1.0");
        }

        ConstantLR {
            total_iters,
            factor,
        }
    }
}

impl Default for ConstantLR {
    fn default() -> Self {
        ConstantLR {
            total_iters: 5,
            factor: 1.0 / 3.0,
        }
    }
}

impl Schedule for ConstantLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if last_epoch == 0 {
            lr * self.factor
        } else if last_epoch == self.total_iters {
            lr * (1.0 / self.factor)
        } else {
            lr
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let one_or_zero = if last_epoch >= self.total_iters {
            1.0
        } else {
            0.0
        };

        base_lr * (self.factor + one_or_zero * (1.0 - self.factor))
    }
}
