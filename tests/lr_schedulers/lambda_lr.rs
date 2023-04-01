extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_schedulers::{LRScheduler, LambdaLR};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [0.05, 0.0475, 0.045125, 0.0428688, 0.0407253];

#[test]
fn valid_lambda_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let gamma: f64 = 0.95;
    let mut scheduler = LRScheduler::new(
        &mut optim,
        LambdaLR {
            base_lr: 0.05,
            lr_lambda: |epoch| gamma.powi(epoch as i32),
        },
    );

    for epoch in 0..5 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch], epsilon = 1e-6);

        scheduler.step();
    }
}
