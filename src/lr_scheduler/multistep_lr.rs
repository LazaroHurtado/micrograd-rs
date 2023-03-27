use super::Schedule;

pub struct MultiStepLR {
    pub milestones: Vec<usize>,
    pub gamma: f64,
}

impl Default for MultiStepLR {
    fn default() -> Self {
        MultiStepLR {
            milestones: vec![],
            gamma: 0.1,
        }
    }
}

impl Schedule for MultiStepLR {
    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if self.milestones.contains(&last_epoch) {
            let occurrences = self
                .milestones
                .iter()
                .filter(|&milestone| *milestone == last_epoch)
                .count() as i32;

            lr * self.gamma.powi(occurrences)
        } else {
            lr
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let power = self
            .milestones
            .iter()
            .filter(|&milestone| *milestone <= last_epoch)
            .count() as i32;

        base_lr * self.gamma.powi(power)
    }
}
