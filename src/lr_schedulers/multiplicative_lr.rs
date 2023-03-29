use super::Schedule;

pub struct MultiplicativeLR<F: Fn(usize) -> f64> {
    pub lr_lambda: F,
}

impl<F> Schedule for MultiplicativeLR<F>
where
    F: Fn(usize) -> f64,
{
    const HAS_CLOSED_FORM: bool = false;

    fn get_lr(&self, lr: f64, last_epoch: usize) -> f64 {
        if last_epoch > 0 {
            lr * (self.lr_lambda)(last_epoch)
        } else {
            lr
        }
    }
}
