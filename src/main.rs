mod activation;
mod layer;
mod mlp;
mod neuron;
mod operation;
mod prelude;
mod tensor;
mod utils;
mod value;

use activation::Activation;
use layer::Layer;
use mlp::MLP;
use prelude::*;

fn main() {
    let mlp: MLP = MLP::new(vec![
        Box::new(Layer::new(3, 4)),
        Box::new(Activation::TanH),
        Box::new(Layer::new(4, 4)),
        Box::new(Activation::TanH),
        Box::new(Layer::new(4, 1)),
        Box::new(Activation::TanH),
    ]);

    let xs = tensor![
        [val!(2.0), val!(3.0), val!(-1.0)],
        [val!(3.0), val!(-1.0), val!(0.5)],
        [val!(0.5), val!(1.0), val!(1.0)],
        [val!(1.0), val!(1.0), val!(-1.0)]
    ];
    let ys = tensor![1, -1, -1, 1];
    let mut ypred: Tensor<Ix1> = Tensor::zeros(4);

    for epoch in 0..20 {
        ypred = mlp.forward_batch::<Ix1>(xs.clone());
        let loss: Value = (ypred.clone() - ys.clone()).map(|val| val.powf(2.0)).sum();

        mlp.zero_grad();
        loss.backward();

        for param in mlp.parameters().iter() {
            let grad = param.grad();
            let mut param_mut = param.0.borrow_mut();
            param_mut.value += grad.value() * -0.1;
        }

        if epoch % 10 == 0 {
            println!("[EPOCH-{:?}] Loss: {:?}", epoch, loss.value());
        }
    }

    if !ypred.is_empty() {
        ypred.iter().for_each(|pred| println!("{:?}", pred.value()));
    }
}
