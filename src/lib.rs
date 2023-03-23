#![crate_name = "micrograd_rs"]

mod ops;

pub mod optimizer;
pub use optimizer as optim;

pub mod lr_scheduler;

pub mod criterion;

pub mod activation;

mod layers;
pub use layers::*;

pub mod prelude;
pub mod utils;

mod tensor;
pub use tensor::Tensor;

mod value;
pub use value::Value;
