mod adam;
mod rmsprop;
mod sgd;

pub use self::adam::Adam;
pub use self::rmsprop::RMSProp;
pub use self::sgd::SGD;
pub use crate::value::Value;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn lr(&self) -> Value;
}
