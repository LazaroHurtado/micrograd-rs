use crate::Value;

mod glorot;
mod he;
mod lecun;

pub use self::glorot::{GlorotNormal, GlorotUniform};
pub use self::he::{HeNormal, HeUniform};
pub use self::lecun::{LecunNormal, LecunUniform};

pub struct Fanning(usize, usize);

impl From<[usize; 2]> for Fanning {
    fn from(val: [usize; 2]) -> Self {
        Fanning(val[0], val[1])
    }
}
impl From<[usize; 1]> for Fanning {
    fn from(val: [usize; 1]) -> Self {
        Fanning(val[0], val[0])
    }
}

pub trait WeightInit {
    fn sample<F: Into<Fanning>>(&self, fanning: F) -> Value;
}
