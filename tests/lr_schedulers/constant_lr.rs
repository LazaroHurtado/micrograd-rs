extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_schedulers::{ConstantLR, LRScheduler};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

#[test]
fn valid_constant_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, factor) = (4, 0.5);
    let mut scheduler = LRScheduler::new(&mut optim, ConstantLR::new(total_iters, factor));

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

    let (total_iters, factor) = (4, 0.5);
    let mut scheduler = LRScheduler::new(&mut optim, ConstantLR::new(total_iters, factor));

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0..=3 => assert_abs_diff_eq!(lr, 0.025, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.05, epsilon = 1e-6),
        }

        scheduler.step_with(epoch + 1);
    }
}

#[test]
#[should_panic]
fn factor_less_than_zero() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, factor) = (4, -0.1);

    LRScheduler::new(&optim, ConstantLR::new(total_iters, factor));
}

#[test]
#[should_panic]
fn factor_greater_than_one() {
    let optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let (total_iters, factor) = (4, 1.1);

    LRScheduler::new(&optim, ConstantLR::new(total_iters, factor));
}
