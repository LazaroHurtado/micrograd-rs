extern crate micrograd_rs;
use approx::assert_abs_diff_eq;
use micrograd_rs::lr_schedulers::{CosineAnnealingLR, LRScheduler};
use micrograd_rs::optim::SGD;
use micrograd_rs::prelude::*;

const VALUES: [f64; 10] = [
    0.05, 0.0452254, 0.0327254, 0.0172746, 0.0047746, 0.0, 0.0047746, 0.0172746, 0.0327254,
    0.0452254,
];

#[test]
fn valid_cosine_annealing_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        CosineAnnealingLR {
            base_lr: 0.05,
            t_max: 5,
            eta_min: 0.0,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch % 10], epsilon = 1e-6);

        scheduler.step();
    }
}

#[test]
fn valid_closed_form_cosine_annealing_lr_scheduler() {
    let mut optim = SGD {
        params: vec![Value::from(0.0)],
        lr: val!(0.05),
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &mut optim,
        CosineAnnealingLR {
            base_lr: 0.05,
            t_max: 5,
            eta_min: 0.0,
        },
    );

    for epoch in 0..100 {
        let lr = scheduler.lr();

        assert_abs_diff_eq!(lr, VALUES[epoch % 10], epsilon = 1e-6);

        scheduler.step_with(epoch + 1);
    }
}
