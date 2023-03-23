extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_scheduler::{LRScheduler, StepLR};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

#[test]
fn valid_step_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        StepLR {
            step_size: 30,
            gamma: 0.1,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0..=29 => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
            30..=59 => assert_abs_diff_eq!(lr, 0.005, epsilon = 1e-6),
            60..=89 => assert_abs_diff_eq!(lr, 0.0005, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.00005, epsilon = 1e-6),
        };

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_step_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        StepLR {
            step_size: 30,
            gamma: 0.1,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0..=29 => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
            30..=59 => assert_abs_diff_eq!(lr, 0.005, epsilon = 1e-6),
            60..=89 => assert_abs_diff_eq!(lr, 0.0005, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.00005, epsilon = 1e-6),
        };

        scheduler.step_with(epoch + 1);
    }
}
