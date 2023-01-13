use crate::prelude::*;
use ndarray::IntoDimension;

//TODO: Implement support for user defined dilation
#[derive(Clone)]
pub struct Filter<D> {
    pub size: D,
    pub stride: D,
    pub dilation: D,
}

impl<D> Filter<D>
where
    D: Dimension,
{
    pub fn new<E: IntoDimension<Dim = D>>(size: E, stride: E) -> Self {
        let mut default_dilation = <D>::zeros(D::NDIM.unwrap());
        for ix in 0..default_dilation.ndim() {
            default_dilation[ix] = 1;
        }

        Filter {
            size: size.into_dimension(),
            stride: stride.into_dimension(),
            dilation: default_dilation,
        }
    }

    pub fn output_shape(&self, input_dim: D) -> D {
        self.output_shape_with_padding(input_dim, <D>::zeros(D::NDIM.unwrap()))
    }

    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    pub fn output_shape_with_padding(&self, input_dim: D, padding: D) -> D {
        let mut output_dim = input_dim.clone();
        let n = output_dim.ndim();

        for ix in 0..n {
            let padding_expansion = 2 * padding[ix];
            let total_kernel_size = self.dilation[ix] * (self.size[ix] - 1);

            let axis =
                ((input_dim[ix] + padding_expansion - total_kernel_size - 1) / self.stride[ix]) + 1;
            output_dim[ix] = axis;
        }

        output_dim
    }
}
