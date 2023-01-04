mod binary_ops;
mod unary_ops;

pub use self::binary_ops::BinaryOps;
pub use self::unary_ops::UnaryOps;

use super::value::Value;

pub trait Op {
    fn variables(&self) -> Vec<Value>;
    fn propagate(&self, source: &Value);
}
