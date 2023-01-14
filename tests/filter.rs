extern crate micrograd_rs;
use micrograd_rs::prelude::*;

#[test]
fn valid_filter_output_size_for_input() {
    let input = (7, 13);

    let (n, m) = (2, 3);
    let (padding, dilation, filter_size, stride) = ((2, 2), 1, (n, m), (1, 2));

    let filter = Filter::new(filter_size, stride, (1, 1));

    let output_h = ((input.0 + 2 * padding.0 - dilation * (filter_size.0 - 1) - 1) / stride.0) + 1;
    let output_w = ((input.1 + 2 * padding.1 - dilation * (filter_size.1 - 1) - 1) / stride.1) + 1;

    assert_eq!(
        filter
            .output_shape_with_padding(Dim(input), Dim(padding))
            .slice(),
        [output_h, output_w]
    );
}

#[test]
fn valid_receptive_field_with_dilation() {
    let (filter_size, stride, dilation) = ((2, 2), (2, 1), (2, 2));
    let filter = Filter::new(filter_size, stride, dilation);

    let input = tensor![[
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.,],
        [17., 18., 19., 20.,],
        [21., 22., 23., 24.,]
    ]];
    let output: Vec<Tensor<Ix3>> = vec![
        tensor![[[1., 3.], [9., 11.]]],
        tensor![[[2., 4.], [10., 12.]]],
        tensor![[[9., 11.], [17., 19.]]],
        tensor![[[10., 12.], [18., 20.]]],
    ];

    assert_eq!(filter.receptive_field(&input), output);
}
