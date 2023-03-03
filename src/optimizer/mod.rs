mod rmsprop;
mod sgd;

pub use self::rmsprop::RMSProp;
pub use self::sgd::SGD;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
