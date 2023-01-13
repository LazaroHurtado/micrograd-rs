extern crate micrograd_rs;
use micrograd_rs::prelude::*;
use micrograd_rs::utils::Kernel;

#[test]
fn returns_all_parameters_in_a_single_kernel() {
    let (in_channels, out_channels) = (3, 13);

    let (n, m) = (3, 8);
    let kernel: Kernel<Ix2, Ix3> =
        Kernel::new(in_channels, out_channels, Filter::new((n, m), (1, 0)));

    let parameters_in_weight_tensor = in_channels * n * m;
    let total_parameters = parameters_in_weight_tensor + 1; // one bias parameter in a kernel

    assert_eq!(kernel.parameters().len(), total_parameters);
}
