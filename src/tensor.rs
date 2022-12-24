use crate::prelude::*;
use ndarray::{ArrayBase, Data, Ix1, Ix2, OwnedRepr};
use std::iter::zip;

pub type Tensor<D> = ArrayBase<OwnedRepr<Value>, D>;

pub trait DotProd<Rhs> {
    type Output;

    fn dot(&self, b: &Rhs) -> Self::Output;
}

impl<S> DotProd<ArrayBase<S, Ix1>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = Value>,
{
    type Output = Value;

    fn dot(&self, b: &ArrayBase<S, Ix1>) -> Self::Output {
        let a = self.view();
        let b = b.view();

        assert!(a.dim() == b.dim());

        zip(a.to_vec(), b.to_vec()).map(|(x, y)| x * y).sum()
    }
}

impl<S> DotProd<ArrayBase<S, Ix2>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = Value>,
{
    type Output = Tensor<Ix1>;

    fn dot(&self, b: &ArrayBase<S, Ix2>) -> Self::Output {
        let a = self.view();
        let b = b.view();

        let (n, _) = b.dim();
        assert!(a.dim() == n);

        let a_t = a.t();

        let mut result = vec![];
        for col in b.columns() {
            result.push(a_t.dot(&col));
        }

        Tensor::<Ix1>::from_vec(result)
    }
}

impl<S> DotProd<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = Value>,
{
    type Output = Tensor<Ix2>;

    fn dot(&self, b: &ArrayBase<S, Ix2>) -> Self::Output {
        let a = self.view();
        let b = b.view();
        let ((m, k), (k2, n)) = (a.dim(), b.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            panic!("Could not multiply");
        }

        let mut result = vec![];

        for row in a.rows() {
            for col in b.columns() {
                result.push(row.dot(&col));
            }
        }

        Tensor::from_shape_vec((m, n), result).unwrap()
    }
}

impl<S> DotProd<ArrayBase<S, Ix1>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = Value>,
{
    type Output = Tensor<Ix2>;

    fn dot(&self, b: &ArrayBase<S, Ix1>) -> Self::Output {
        let a = self.view();
        let b = b.view();

        let (m, k) = a.dim();
        let n = b.dim();

        if k != 1 || m.checked_mul(n).is_none() {
            panic!("Could not multiply");
        }

        let b_elems = b.to_vec();
        let mut result = vec![];

        for row in a.rows() {
            let x = row.get(0).unwrap();
            for elem in b_elems.iter() {
                result.push(elem * x);
            }
        }

        Tensor::from_shape_vec((m, n), result).unwrap()
    }
}
