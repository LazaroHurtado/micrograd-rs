use super::Schedule;

pub struct LambdaLR<F: Fn(usize) -> f64> {
    pub base_lr: f64,
    pub lr_lambda: F,
}

impl<F> Schedule for LambdaLR<F>
where
    F: Fn(usize) -> f64,
{
    const HAS_CLOSED_FORM: bool = false;

    fn get_lr(&self, _: f64, last_epoch: usize) -> f64 {
        self.base_lr * (self.lr_lambda)(last_epoch)
    }
}
