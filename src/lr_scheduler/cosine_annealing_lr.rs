use super::Schedule;
use std::f64::consts::PI;

pub struct CosineAnnealingLR {
    pub base_lr: f64,
    pub t_max: usize,
    pub eta_min: f64,
}

impl Default for CosineAnnealingLR {
    fn default() -> Self {
        CosineAnnealingLR {
            base_lr: 0.05,
            t_max: 5,
            eta_min: 0.0,
        }
    }
}

impl Schedule for CosineAnnealingLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if last_epoch == 0 {
            return lr;
        }

        let (epoch_minus_one, t_max) = ((last_epoch as i32) - 1, self.t_max as i32);

        let lr_range = self.base_lr - self.eta_min;
        let wavelen_percent = PI * (last_epoch as f64) / (t_max as f64);

        if last_epoch == 1 {
            self.eta_min + lr_range * (1.0 + wavelen_percent.cos()) / 2.0
        } else if (epoch_minus_one - t_max) % (2 * t_max) == 0 {
            let wavelen_step = PI / (t_max as f64);

            lr + lr_range * (1.0 - wavelen_step.cos()) / 2.0
        } else {
            let offset_wavelen_percent = PI * (epoch_minus_one as f64) / (t_max as f64);
            let cos_annealing =
                (1.0 + wavelen_percent.cos()) / (1.0 + offset_wavelen_percent.cos());

            self.eta_min + cos_annealing * (lr - self.eta_min)
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let lr_range = base_lr - self.eta_min;
        let wavelen_percent = PI * (last_epoch as f64) / (self.t_max as f64);

        self.eta_min + lr_range * (1.0 + wavelen_percent.cos()) / 2.0
    }
}
