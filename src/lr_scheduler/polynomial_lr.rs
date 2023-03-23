use super::Schedule;

pub struct PolynomialLR {
    pub total_iters: usize,
    pub power: i32,
}

impl Default for PolynomialLR {
    fn default() -> Self {
        PolynomialLR {
            total_iters: 5,
            power: 1,
        }
    }
}

impl Schedule for PolynomialLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if (last_epoch == 0) || (last_epoch > self.total_iters) {
            lr
        } else {
            let total_iters = self.total_iters as f64;
            let base = (1.0 - (last_epoch as f64) / total_iters)
                / (1.0 - ((last_epoch - 1) as f64) / total_iters);

            lr * base.powi(self.power)
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let percent_iters = (last_epoch.min(self.total_iters) as f64) / (self.total_iters as f64);

        base_lr * (1.0 - percent_iters).powi(self.power)
    }
}
