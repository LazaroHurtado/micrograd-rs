extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::{Activation, Criterion, Linear, Module, Reduction, Sequential};

fn main() {
    let feedforward = sequential!(
        Ix1,
        [
            Linear::new(3, 4),
            Activation::Tanh,
            Linear::new(4, 4),
            Activation::Tanh,
            Linear::new(4, 1),
            Activation::Tanh
        ]
    );

    let xs = tensor![
        [val!(2.0), val!(3.0), val!(-1.0)],
        [val!(3.0), val!(-1.0), val!(0.5)],
        [val!(0.5), val!(1.0), val!(1.0)],
        [val!(1.0), val!(1.0), val!(-1.0)]
    ];
    let ys = tensor![1, -1, -1, 1];
    let mut ypred: Tensor<Ix1> = Tensor::zeros(4);

    let criterion = Criterion::MSE;

    for epoch in 0..=20 {
        ypred = feedforward
            .forward_batch(xs.clone())
            .to_shape(4)
            .unwrap()
            .to_owned();
        let loss: Value = criterion.loss(Reduction::Sum, ypred.clone(), ys.clone());

        feedforward.zero_grad();
        loss.backward();

        for param in feedforward.parameters().iter_mut() {
            let grad = param.grad().unwrap();
            *param.value_mut() += grad.value() * -0.1;
        }

        if epoch % 10 == 0 {
            println!("[EPOCH-{:?}] Loss: {:?}", epoch, loss.value());
        }
    }

    if !ypred.is_empty() {
        ypred.iter().for_each(|pred| println!("{:?}", pred.value()));
    }
}
