use super::Schedule;

pub struct LinearLR {
    pub total_iters: usize,
    pub start_factor: f64,
    pub end_factor: f64,
}

impl LinearLR {
    pub fn new(total_iters: usize, start_factor: f64, end_factor: f64) -> Self {
        if start_factor <= 0.0 || start_factor > 1.0 {
            panic!("Starting factor expected to be greater than 0.0 and less or equal to 1.0");
        }
        if !(0.0..=1.0).contains(&end_factor) {
            panic!("Ending factor expected to be between 0.0 and 1.0");
        }

        LinearLR {
            total_iters,
            start_factor,
            end_factor,
        }
    }
}

impl Default for LinearLR {
    fn default() -> Self {
        LinearLR {
            total_iters: 5,
            start_factor: 1.0 / 3.0,
            end_factor: 1.0,
        }
    }
}

impl Schedule for LinearLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if last_epoch == 0 {
            lr * self.start_factor
        } else if last_epoch > self.total_iters {
            lr
        } else {
            let factor_range = self.end_factor - self.start_factor;
            let denominator = self.start_factor * (self.total_iters as f64)
                + factor_range * ((last_epoch - 1) as f64);

            lr * (1.0 + factor_range / denominator)
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let (last_epoch, total_iters) = (last_epoch as f64, self.total_iters as f64);

        let factor_range = self.end_factor - self.start_factor;
        let percent_iters = last_epoch.min(total_iters) / total_iters;

        base_lr * (self.start_factor + factor_range * percent_iters)
    }
}
