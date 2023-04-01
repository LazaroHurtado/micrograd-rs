#![crate_name = "micrograd_rs"]

mod ops;

pub mod optimizers;
pub use optimizers as optim;

pub mod lr_schedulers;

pub mod criterions;

pub mod activations;

mod layers;
pub use layers::*;

pub mod prelude;
pub mod utils;

mod tensor;
pub use tensor::Tensor;

mod value;
pub use value::Value;
