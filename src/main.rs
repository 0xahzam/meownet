mod engine;

fn main() {
    //Inputs x1, x2
    let mut x1 = engine::Value::new(2.0);
    let mut x2 = engine::Value::new(0.0);

    // Inputs w1, w2
    let mut w1 = engine::Value::new(-3.0);
    let mut w2 = engine::Value::new(1.0);

    // Bias of the neuron
    let mut b = engine::Value::new(6.8813735870195432);

    // x1*w1 + x2*w2 + b
    let mut x1w1 = &x1 * &w1;
    let mut x2w2 = &x2 * &w2;
    let mut x1w1x2w2 = &x1w1 + &x2w2;
    let mut n = &x1w1x2w2 + &b;
    let mut o = n.tanh();

    println!("\nVALUES\n");
    println!("o             :: {}", o.data);
    println!("n             :: {}", n.data);
    println!("b             :: {}", b.data);
    println!("x1w1x2w2      :: {}", x1w1x2w2.data);
    println!("x1w1          :: {}", x1w1.data);
    println!("x2w2          :: {}", x2w2.data);
    println!("x1            :: {}", x1.data);
    println!("w1            :: {}", w1.data);
    println!("x2            :: {}", x2.data);
    println!("w2            :: {}", w2.data);

    // Calculating backwardpass manually
    o.grad = 1.0;
    n.backward(&o);
    b.backward(&n);
    x1w1x2w2.backward(&n);
    x1w1.backward(&x1w1x2w2);
    x2w2.backward(&x1w1x2w2);
    w1.backward(&x1w1);
    x1.backward(&x1w1);
    w2.backward(&x2w2);
    x2.backward(&x2w2);

    println!("\nGRADIENTS\n");
    println!("o grd         :: {}", o.grad);
    println!("n grd         :: {}", n.grad);
    println!("b grd         :: {}", b.grad);
    println!("x1w1x2w2 grd  :: {}", x1w1x2w2.grad);
    println!("x1w1 grd      :: {}", x1w1.grad);
    println!("x2w2 grd      :: {}", x2w2.grad);
    println!("x1 grd        :: {}", x1.grad);
    println!("w1 grd        :: {}", w1.grad);
    println!("x2 grd        :: {}", x2.grad);
    println!("w2 grd        :: {}", w2.grad);
}
