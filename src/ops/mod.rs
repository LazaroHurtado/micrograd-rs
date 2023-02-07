mod binary_ops;
mod unary_ops;

pub use self::binary_ops::BinaryOps;
pub use self::unary_ops::UnaryOps;

use super::value::Value;

pub trait Op {
    fn into_inner(self) -> Vec<Value>;
    fn variables(&self) -> Vec<&Value>;
    fn propagate(&self, source: &Value);
}

#[derive(Default)]
pub enum Ops {
    Binary(BinaryOps),
    Unary(UnaryOps),
    #[default]
    NoOp,
}

impl Op for Ops {
    fn into_inner(self) -> Vec<Value> {
        match self {
            Self::Binary(bin_ops) => bin_ops.into_inner(),
            Self::Unary(unary_ops) => unary_ops.into_inner(),
            Self::NoOp => vec![],
        }
    }

    fn variables(&self) -> Vec<&Value> {
        match self {
            Self::Binary(bin_ops) => bin_ops.variables(),
            Self::Unary(unary_ops) => unary_ops.variables(),
            Self::NoOp => vec![],
        }
    }

    fn propagate(&self, source: &Value) {
        match self {
            Self::Binary(bin_ops) => bin_ops.propagate(source),
            Self::Unary(unary_ops) => unary_ops.propagate(source),
            Self::NoOp => (),
        }
    }
}

macro_rules! impl_into_ops {
    [$(($op: ty, $varient: ident)),*] => {
        $(impl From<$op> for Ops {
            fn from(bin_ops: $op) -> Self {
                Ops::$varient(bin_ops)
            }
        })*
    };
}
impl_into_ops![(BinaryOps, Binary), (UnaryOps, Unary)];
