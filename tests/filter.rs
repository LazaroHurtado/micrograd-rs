extern crate micrograd_rs;
use micrograd_rs::prelude::*;

#[test]
fn valid_kernel_output_size_for_input() {
    let input = (7, 13);

    let (n, m) = (2, 3);
    let (padding, dilation, kernel_size, stride) = ((2, 2), 1, (n, m), (1, 2));

    let filter = Filter::new(kernel_size, stride);

    let output_h = ((input.0 + 2 * padding.0 - dilation * (kernel_size.0 - 1) - 1) / stride.0) + 1;
    let output_w = ((input.1 + 2 * padding.1 - dilation * (kernel_size.1 - 1) - 1) / stride.1) + 1;

    assert_eq!(
        filter
            .output_shape_with_padding(Dim(input), Dim(padding))
            .slice(),
        [output_h, output_w]
    );
}
