extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_schedulers::{ExponentialLR, LRScheduler};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

const VALUES: [f64; 5] = [0.05, 0.04, 0.032, 0.0256, 0.02048];

#[test]
fn valid_exponential_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(&mut optim, ExponentialLR { gamma: 0.8 });

    for epoch in 0..5 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch], epsilon = 1e-6);

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_exponential_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(&mut optim, ExponentialLR { gamma: 0.8 });

    for epoch in 0..5 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch], epsilon = 1e-6);

        scheduler.step_with(epoch + 1);
    }
}
