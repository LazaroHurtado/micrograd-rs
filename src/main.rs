mod activation;
mod layer;
mod mlp;
mod neuron;
mod operation;
mod value;

use activation::Activation;
use layer::Layer;
use mlp::MLP;
use std::iter::zip;
use value::Value;

// fn value_test() {
//     let x = Value::from(3.0);
//     println!("f(x) = x^3");
//
//     let y = &x.powf(3.0);
//     println!("f(3) = 3^3 = {:?}", y.value());
//     y.backward();
//
//     let dy = &x.grad();
//     println!("f'(3) = 3*(3)^2 = {:?}", dy.value());
//     x.zero_grad();
//     dy.backward();
//
//     let d2y = &x.grad();
//     println!("f''(3) = 6*(3) {:?}", d2y.value());
//     x.zero_grad();
//     d2y.backward();
//
//     let d3y = &x.grad();
//     println!("f'''(3) = 6 = {:?}", d3y.value());
//     x.zero_grad();
//     d3y.backward();
//
//     let d4y = &x.grad();
//     println!("f''''(3) = 0 = {:?}", d4y.value());
//     x.zero_grad();
// }

fn main() {
    // value_test();

    let mlp: MLP = MLP::new(vec![
        Box::new(Layer::new(3, 4)),
        Box::new(Activation::TanH),
        Box::new(Layer::new(4, 4)),
        Box::new(Activation::TanH),
        Box::new(Layer::new(4, 1)),
        Box::new(Activation::TanH),
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
            let grad = param.grad();
            let mut param_mut = param.0.borrow_mut();
            param_mut.value += grad.value() * -0.1;
        }

        println!("[EPOCH-{:?}] Loss: {:?}", epoch, loss.value());
    }

    if !ypred.is_empty() {
        ypred.iter().for_each(|pred| println!("{:?}", pred.value()));
    }
}
