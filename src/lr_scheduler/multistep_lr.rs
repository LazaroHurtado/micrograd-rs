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
        let milestone = self
            .milestones
            .iter()
            .position(|&epoch| epoch == last_epoch);

        match milestone {
            None => lr,
            Some(_) => {
                let power =
                    self.milestones.iter().fold(
                        0,
                        |acc, &epoch| if epoch == last_epoch { acc + 1 } else { acc },
                    );

                lr * self.gamma.powi(power)
            }
        }
    }

    fn get_closed_form_lr(&self, base_lr: f64, last_epoch: usize) -> f64 {
        let power = self.milestones.iter().fold(
            0,
            |acc, &epoch| if epoch <= last_epoch { acc + 1 } else { acc },
        );

        base_lr * self.gamma.powi(power)
    }
}
