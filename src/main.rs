mod operation;
mod value;

use value::Value;

fn main() {
    let mut a = Value::new(2.0);
    let mut b = Value::new(-3.0);

    let mut e = &mut a * &mut b;
    let mut c = Value::new(10.0);

    let mut d = &mut e + &mut c;
    let mut f = Value::new(-2.0);

    let mut l = &mut d * &mut f;
    l.backward(true);

    println!("{:?}", l.params());
    println!("{:?}", l);
}
