use ndarray::IntoDimension;

use super::PoolingFn;
use crate::prelude::*;

pub struct MaxPool<D> {
    pub size: D,
    pub stride: D,
    pub padding: D,
    pub dilation: D,
}

impl<D> Default for MaxPool<D>
where
    D: Dimension,
{
    fn default() -> Self {
        let n = D::NDIM.unwrap();

        let zeros = D::zeros(n);
        let mut ones = D::zeros(n);
        ones.slice_mut().fill(1);

        Self {
            size: ones.clone(),
            stride: ones.clone(),
            padding: zeros,
            dilation: ones,
        }
    }
}

impl<D> MaxPool<D>
where
    D: Dimension,
{
    pub fn new<J: IntoDimension<Dim = D>>(size: J, stride: J, padding: J, dilation: J) -> Self {
        Self {
            size: size.into_dimension(),
            padding: padding.into_dimension(),
            stride: stride.into_dimension(),
            dilation: dilation.into_dimension(),
        }
    }

    fn max_pooling(&self, window: Tensor<D>) -> Value {
        window.into_iter().reduce(Value::max).unwrap()
    }
}

impl<D> PoolingFn<D> for MaxPool<D>
where
    D: Dimension,
{
    fn pool_name(&self) -> String {
        String::from("MaxPooling")
    }

    fn pool(&self, input: Tensor<D>) -> Tensor<D> {
        let mut pooled_input = vec![];

        for window in input.windows_with_stride(self.size.clone(), self.stride.clone()) {
            let pooled_window = self.max_pooling(window.into_owned());
            pooled_input.push(pooled_window);
        }

        let output_dim = self.output_shape(&input.raw_dim());
        Tensor::from_shape_vec(output_dim, pooled_input).unwrap()
    }

    fn output_shape<E: Dimension>(&self, input_dim: &E) -> E {
        let mut output_dim = input_dim.clone();
        let n = input_dim.ndim();
        let pool_n = D::NDIM.unwrap();
        let diff_n = n - pool_n;

        for ix in 0..pool_n {
            let padding_expansion = 2 * self.padding[ix];
            let total_filter_size = self.dilation[ix] * (self.size[ix] - 1);

            let axis = ((input_dim[diff_n + ix] + padding_expansion - total_filter_size - 1)
                / self.stride[ix])
                + 1;
            output_dim[diff_n + ix] = axis;
        }

        output_dim
    }
}
