extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_scheduler::{LRScheduler, MultiplicativeLR};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [0.05, 0.04375, 0.0413194, 0.0400282, 0.0392276];

#[test]
fn valid_multiplicative_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        MultiplicativeLR {
            lr_lambda: |epoch| 1.0 - 1.0 / (2.0 * (epoch as f64 + 1.0).powi(2)),
        },
    );

    for epoch in 0..5 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch], epsilon = 1e-6);

        scheduler.step();
    }
}
