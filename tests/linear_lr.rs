extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_scheduler::{LRScheduler, LinearLR};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

#[test]
fn valid_linear_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 0.5, 1.0);

    let mut scheduler = LRScheduler::new(
        &mut optim,
        LinearLR::new(total_iters, start_factor, end_factor),
    );

    for epoch in 0..10 {
        let lr = scheduler.lr();

        match epoch {
            0 => assert_abs_diff_eq!(lr, 0.025, epsilon = 1e-6),
            1 => assert_abs_diff_eq!(lr, 0.03125, epsilon = 1e-6),
            2 => assert_abs_diff_eq!(lr, 0.0375, epsilon = 1e-6),
            3 => assert_abs_diff_eq!(lr, 0.04375, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
        }

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_linear_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 0.5, 1.0);

    let mut scheduler = LRScheduler::new(
        &mut optim,
        LinearLR::new(total_iters, start_factor, end_factor),
    );

    for epoch in 0..10 {
        let lr = scheduler.lr();

        match epoch {
            0 => assert_abs_diff_eq!(lr, 0.025, epsilon = 1e-6),
            1 => assert_abs_diff_eq!(lr, 0.03125, epsilon = 1e-6),
            2 => assert_abs_diff_eq!(lr, 0.0375, epsilon = 1e-6),
            3 => assert_abs_diff_eq!(lr, 0.04375, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
        }

        scheduler.step_with(epoch + 1);
    }
}

#[test]
#[should_panic]
fn start_factor_equals_zero() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 0.0, 1.0);

    LRScheduler::new(&optim, LinearLR::new(total_iters, start_factor, end_factor));
}

#[test]
#[should_panic]
fn start_factor_greater_than_one() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 1.1, 1.0);

    LRScheduler::new(&optim, LinearLR::new(total_iters, start_factor, end_factor));
}

#[test]
#[should_panic]
fn end_factor_less_than_zero() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 0.5, -0.1);

    LRScheduler::new(&optim, LinearLR::new(total_iters, start_factor, end_factor));
}

#[test]
#[should_panic]
fn end_factor_greater_than_one() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, start_factor, end_factor) = (4, 0.5, 1.1);

    LRScheduler::new(&optim, LinearLR::new(total_iters, start_factor, end_factor));
}
