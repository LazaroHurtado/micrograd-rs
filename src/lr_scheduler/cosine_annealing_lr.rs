use super::Schedule;

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

        let pi = std::f64::consts::PI;
        let lr_range = self.base_lr - self.eta_min;
        let wavelen_percent = pi * (last_epoch as f64) / (self.t_max as f64);

        if last_epoch == 1 {
            self.eta_min + lr_range * (1.0 + wavelen_percent.cos()) / 2.0
        } else if (last_epoch as i32 - self.t_max as i32 - 1) % (2 * self.t_max as i32) == 0 {
            let wavelen_step = pi / (self.t_max as f64);

            lr + lr_range * (1.0 - wavelen_step.cos()) / 2.0
        } else {
            let offset_wavelen_percent = pi * ((last_epoch - 1) as f64) / (self.t_max as f64);
            let cos_annealing =
                (1.0 + wavelen_percent.cos()) / (1.0 + offset_wavelen_percent.cos());

            self.eta_min + cos_annealing * (lr - self.eta_min)
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let pi = std::f64::consts::PI;
        let lr_range = base_lr - self.eta_min;
        let wavelen_percent = pi * (last_epoch as f64) / (self.t_max as f64);

        self.eta_min + lr_range * (1.0 + wavelen_percent.cos()) / 2.0
    }
}
