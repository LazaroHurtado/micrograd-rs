mod layer;
mod mlp;
mod neuron;
mod operation;
mod value;

use layer::Layer;
use mlp::MLP;
use neuron::Activation;
use std::iter::zip;
use value::Value;

// fn value_test() {
//     let x1 = Value::new(2.0);
//     let x2 = Value::new(0.0);
//
//     let w1 = Value::new(-3.0);
//     let w2 = Value::new(1.0);
//
//     let b = Value::new(6.8813735870195432);
//
//     let x1w1 = &x1 * &w1;
//     let x2w2 = &x2 * &w2;
//
//     let x1w1x2w2 = &x1w1 + &x2w2;
//
//     let n = &x1w1x2w2 + &b;
//     println!("{:?}", n);
// }

fn main() {
    let mlp: MLP = MLP::new(vec![
        Layer::new(3, 4, Some(Activation::TanH)),
        Layer::new(4, 4, Some(Activation::TanH)),
        Layer::new(4, 1, Some(Activation::TanH)),
    ]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];
    let mut ypred: Vec<Value> = Vec::new();

    for epoch in 0..20 {
        ypred = xs
            .iter()
            .map(|x| mlp.forward(x.clone()).pop().unwrap())
            .collect();

        let loss: Value = zip(ys.clone(), ypred.clone())
            .map(|(y, y_hat)| {
                let error = y_hat - Value::new(y);
                error.powf(2.0)
            })
            .sum();

        mlp.zero_grad();
        loss.backward();

        for param in mlp.parameters().iter() {
            let mut param_mut = param.0.borrow_mut();
            param_mut.value += -0.1 * param_mut.grad;
        }

        println!("[EPOCH-{:?}] Loss: {:?}", epoch, loss.value());
    }

    if !ypred.is_empty() {
        ypred.iter().for_each(|pred| println!("{:?}", pred.value()));
    }
}
