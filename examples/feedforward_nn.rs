extern crate micrograd_rs;
use micrograd_rs::activation as Activation;
use micrograd_rs::criterion::{Criterion, Reduction, MSE};
use micrograd_rs::lr_scheduler::{ConstantLR, LRScheduler};
use micrograd_rs::optim::{Optimizer, SGD};
use micrograd_rs::prelude::*;
use micrograd_rs::{Layer, Linear, Sequential};

fn main() {
    let model = sequential!(
        Ix1,
        [
            Linear::new("fc1", 3, 4),
            Activation::Tanh,
            Linear::new("fc2", 4, 4),
            Activation::Tanh,
            Linear::new("fc3", 4, 1),
            Activation::Tanh
        ]
    );

    let xs = tensor![
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ];
    let ys = tensor!([[1.], [-1.], [-1.], [1.]], requires_grad = false);
    let mut ypred: Tensor<Ix2> = Tensor::zeros((4, 1));

    let mut optimizer = SGD {
        params: model.parameters().into_raw_vec(),
        lr: val!(0.1),
        momentum: 0.3,
        ..Default::default()
    };

    let mut scheduler = LRScheduler::new(
        &optimizer,
        ConstantLR {
            total_iters: 4,
            factor: 0.8,
        },
    );

    for epoch in 0..20 {
        ypred = model.forward_batch(&xs);
        let loss: Value = MSE::loss(Reduction::Sum, &ypred, &ys);

        optimizer.zero_grad();
        loss.backward();

        optimizer.step();
        scheduler.step();

        if epoch % 10 == 0 {
            println!("[EPOCH-{:?}] Loss: {:?}", epoch, loss.value());
        }
    }

    if !ypred.is_empty() {
        ypred.iter().for_each(|pred| println!("{:?}", pred.value()));
    }
}
