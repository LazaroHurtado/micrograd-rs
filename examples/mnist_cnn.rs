extern crate micrograd_rs;
use micrograd_rs::activations as Activation;
use micrograd_rs::pooling::MaxPool;
use micrograd_rs::prelude::*;
use micrograd_rs::{Conv2D, Layer, Linear, Model, Sequential};

use mnist::*;
use rand::Rng;
use std::process::Command;

struct CNN {
    cnn: Sequential<Ix4>,
    linear: Sequential<Ix1>,
}

impl Layer<Ix4, Ix1> for CNN {
    fn forward(&self, input: &Tensor<Ix4>) -> Tensor<Ix1> {
        let cnn_out = self.cnn.forward(input);
        let size = cnn_out.len();
        let flattend = cnn_out.into_shape(size).unwrap();
        self.linear.forward(&flattend)
    }

    fn name(&self) -> String {
        "mnist_cnn_model".to_string()
    }
}

fn main() {
    Command::new("/bin/sh")
        .arg("-c")
        .arg("./examples/download_mnist_dataset.sh")
        .output()
        .expect("Could not download MNIST dataset:");

    let mnist = MnistBuilder::new()
        .label_format_digit()
        .test_set_length(500)
        .base_path("./examples/mnist_dataset")
        .finalize();
    let Mnist {
        tst_img, tst_lbl, ..
    } = mnist;

    let test_data = Array4::from_shape_vec((500, 1, 28, 28), tst_img)
        .expect("Error converting images to Array4 struct:")
        .map(|x| Value::from((*x as f32) / 256.0));

    let test_labels = Array2::from_shape_vec((500, 1), tst_lbl)
        .expect("Error converting labels to Array2 struct:");

    let mut model = CNN {
        cnn: sequential!(
            Ix4,
            [
                Conv2D::new("cnn1", 1, 16, (5, 5), (0, 0), (1, 1), (1, 1)),
                Activation::ReLU,
                MaxPool::new((2, 2), (2, 2), (0, 0), (1, 1)),
                Conv2D::new("cnn2", 16, 32, (5, 5), (0, 0), (1, 1), (1, 1)),
                Activation::ReLU,
                MaxPool::new((2, 2), (2, 2), (0, 0), (1, 1))
            ]
        ),
        linear: sequential!(
            Ix1,
            [Linear::new("fc1", 32 * 4 * 4, 10), Activation::Softmax(0)]
        ),
    };

    model
        .cnn
        .load_state_dict("./examples/mnist_cnn_model.pickle");
    model
        .linear
        .load_state_dict("./examples/mnist_cnn_model.pickle");

    let id = rand::thread_rng().gen_range(0..500);
    let x = test_data.slice(s![id, .., .., ..]).insert_axis(Axis(0));
    let y = test_labels.slice(s![id, 0]).to_owned().into_raw_vec();

    let label = *y.first().unwrap() as usize;

    let outputs = model.forward(&x.to_owned()).mapv(|v| v.value());
    let (predicted, probability) = outputs
        .into_iter()
        .enumerate()
        .reduce(|(i, u), (j, v)| match u >= v {
            true => (i, u),
            false => (j, v),
        })
        .unwrap();

    println!("Actual Label: {:?}", label);
    println!(
        "Predicted {:?} with {:.2}% probability",
        predicted,
        probability * 100.0
    );
}
