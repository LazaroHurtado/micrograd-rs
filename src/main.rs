mod operation;
mod value;

use value::Value;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);

    let e = &a * &b;
    let c = Value::new(10.0);

    let d = &e + &c;
    let f = Value::new(-2.0);

    let l = &d * &f;
    l.backward(true);

    println!("a grad: {:?}", a.0.borrow().grad);
    println!("b grad: {:?}", b.0.borrow().grad);
    println!("e grad: {:?}", e.0.borrow().grad);
    println!("c grad: {:?}", c.0.borrow().grad);
    println!("d grad: {:?}", d.0.borrow().grad);
    println!("f grad: {:?}", f.0.borrow().grad);
    println!("l grad {:?}", l.0.borrow().grad);
}
