#![crate_name = "micrograd_rs"]

mod activation;
pub use activation::*;

mod modules;
pub use modules::*;

mod criterion;
pub use criterion::*;

mod ops;

pub mod prelude;
pub mod utils;

mod tensor;
pub use tensor::Tensor;

mod value;
pub use value::Value;
