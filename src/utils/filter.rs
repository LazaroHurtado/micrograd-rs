use crate::prelude::*;
use ndarray::{IntoDimension, Slice};

#[derive(Clone)]
pub struct Filter<D> {
    pub size: D,
    pub stride: D,
    pub dilation: D,
}

impl<D> Default for Filter<D>
where
    D: Dimension,
{
    fn default() -> Self {
        let mut one_across_all_dims = <D>::zeros(D::NDIM.unwrap());
        for ix in 0..D::NDIM.unwrap() {
            one_across_all_dims[ix] = 1;
        }

        Filter {
            size: one_across_all_dims.clone(),
            stride: one_across_all_dims.clone(),
            dilation: one_across_all_dims.clone(),
        }
    }
}

impl<D, E> Filter<D>
where
    D: Dimension<Larger = E>,
    E: Dimension,
{
    pub fn new<J: IntoDimension<Dim = D>>(size: J, stride: J, dilation: J) -> Self {
        Filter {
            size: size.into_dimension(),
            stride: stride.into_dimension(),
            dilation: dilation.into_dimension(),
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
            let total_filter_size = self.dilation[ix] * (self.size[ix] - 1);

            let axis =
                ((input_dim[ix] + padding_expansion - total_filter_size - 1) / self.stride[ix]) + 1;
            output_dim[ix] = axis;
        }

        output_dim
    }

    fn filter_size(&self) -> D {
        let mut size = self.size.clone();

        for ix in 0..size.ndim() {
            size[ix] += (self.dilation[ix] - 1) * (size[ix] - 1);
        }

        size
    }

    fn dilate_filter(&self, filter: &Tensor<E>) -> Tensor<E> {
        let dilation = self.dilation.insert_axis(Axis(0));
        let dilated_filter =
            filter.slice_each_axis(|ax| Slice::new(0, None, dilation[ax.axis.index()] as isize));

        dilated_filter.to_owned()
    }

    pub fn receptive_field(&self, input: &Tensor<E>) -> Vec<Tensor<E>> {
        let mut filters = vec![];

        let dilated_size = self.filter_size();
        let size = dilated_size.insert_axis(Axis(0));
        let stride = self.stride.insert_axis(Axis(0));

        for filter in input.windows_with_stride(size, stride) {
            let dilated_filter = self.dilate_filter(&filter.to_owned());
            filters.push(dilated_filter);
        }

        filters
    }
}
