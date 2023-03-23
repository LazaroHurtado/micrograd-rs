extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_scheduler::{LRScheduler, PolynomialLR};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

#[test]
fn valid_polynomial_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.001),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        PolynomialLR {
            total_iters: 4,
            power: 1,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0 => assert_abs_diff_eq!(lr, 0.001, epsilon = 1e-6),
            1 => assert_abs_diff_eq!(lr, 0.00075, epsilon = 1e-6),
            2 => assert_abs_diff_eq!(lr, 0.0005, epsilon = 1e-6),
            3 => assert_abs_diff_eq!(lr, 0.00025, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.0, epsilon = 1e-6),
        }

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_polynomial_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.001),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        PolynomialLR {
            total_iters: 4,
            power: 1,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        match epoch {
            0 => assert_abs_diff_eq!(lr, 0.001, epsilon = 1e-6),
            1 => assert_abs_diff_eq!(lr, 0.00075, epsilon = 1e-6),
            2 => assert_abs_diff_eq!(lr, 0.0005, epsilon = 1e-6),
            3 => assert_abs_diff_eq!(lr, 0.00025, epsilon = 1e-6),
            _ => assert_abs_diff_eq!(lr, 0.0, epsilon = 1e-6),
        }

        scheduler.step_with(epoch + 1);
    }
}
