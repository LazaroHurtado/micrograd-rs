use super::Schedule;

pub struct ExponentialLR {
    pub gamma: f64,
}

impl Schedule for ExponentialLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if last_epoch == 0 {
            lr
        } else {
            lr * self.gamma
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        base_lr * (self.gamma).powi(last_epoch as i32)
    }
}
