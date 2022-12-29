extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::{Conv3D, Filter, Kernel, Module};

#[test]
fn returns_all_parameters_in_a_single_kernel() {
    let (in_channels, out_channels) = (3, 13);

    let (n, m) = (3, 8);
    let kernel: Kernel<Ix2, Ix3> = Kernel::new(in_channels, out_channels, (n, m), (1, 0));

    let parameters_in_weight_tensor = in_channels * n * m;
    let total_parameters = parameters_in_weight_tensor + 1; // one bias parameter in a kernel

    assert_eq!(kernel.parameters().len(), total_parameters);
}

#[test]
fn returns_all_parameters_in_each_kernel() {
    let (in_channels, out_channels) = (3, 13);

    let (n, m, k) = (3, 8, 4);
    let conv3d = Conv3D::new(in_channels, out_channels, (n, m, k), (1, 0, 0));

    let parameters_per_kernel = in_channels * n * m * k;
    let total_kernel_parameters = out_channels * parameters_per_kernel;
    let total_parameters = total_kernel_parameters + out_channels; // one bias parameter per
                                                                   // kernel

    assert_eq!(conv3d.parameters().len(), total_parameters);
}

#[test]
fn valid_kernel_output_size_for_input() {
    let (input_h, input_w) = (7, 13);

    let (n, m) = (2, 3);
    let (padding, dilation, kernel_size, stride) = (0, 1, (n, m), (1, 2));

    let filter = Filter::new(kernel_size, stride);

    let output_h = ((input_h + 2 * padding - dilation * (kernel_size.0 - 1) - 1) / stride.0) + 1;
    let output_w = ((input_w + 2 * padding - dilation * (kernel_size.1 - 1) - 1) / stride.1) + 1;

    assert_eq!(
        filter.output_shape(Dim((input_h, input_w))).slice(),
        [output_h, output_w]
    );
}
