use super::Schedule;

pub struct StepLR {
    pub step_size: usize,
    pub gamma: f64,
}

impl Default for StepLR {
    fn default() -> Self {
        StepLR {
            step_size: 5,
            gamma: 0.1,
        }
    }
}

impl Schedule for StepLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if (last_epoch == 0) || (last_epoch % self.step_size != 0) {
            lr
        } else {
            lr * self.gamma
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let power = (last_epoch / self.step_size) as i32;

        base_lr * (self.gamma).powi(power)
    }
}
