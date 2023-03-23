extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_scheduler::{ConstantLR, LRScheduler};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

#[test]
fn valid_constant_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        ConstantLR {
            total_iters: 4,
            factor: 0.5,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0..=3 => assert_abs_diff_eq!(lr, 0.025, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
        }

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_constant_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        ConstantLR {
            total_iters: 4,
            factor: 0.5,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0..=3 => assert_abs_diff_eq!(lr, 0.025, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
        }

        scheduler.step_with(epoch + 1);
    }
}
