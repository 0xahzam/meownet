use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// Initializing Value struct
pub(crate) struct Value {
    pub data: f64,
    pub op: String,
    pub grad: f64,
}

// Equivalent of __init__ in Python
impl Value {
    pub fn new(data: f64) -> Self {
        Value {
            data,
            op: String::new(),
            grad: 0.0,
        }
    }

    // Calculate tanh
    pub fn tanh(&self) -> Value {
        Value {
            data: self.data.tanh(),
            grad: self.grad,
            op: "tanh".to_string(),
        }
    }

    // Calculate ReLu
    pub fn relu(&self) -> Value {
        Value {
            data: if self.data > 0.0 { self.data } else { 0.0 },
            grad: 0.0,
            op: "relu".to_string(),
        }
    }

    // Calculate backwardpass for different operations
    pub fn backward(&mut self, grad: &Value) {
        match grad.op.as_str() {
            "+" => {
                if self.data == grad.data - self.data {
                    self.grad += 2.0 * grad.grad
                } else {
                    self.grad += grad.grad;
                }
            }

            "*" => {
                fn safe_divide(num: f64, den: f64) -> f64 {
                    if den.abs() < f64::EPSILON {
                        1.0
                    } else {
                        num / den
                    }
                }

                if self.data == grad.data / self.data {
                    self.grad += 2.0 * safe_divide(grad.data, self.data);
                } else {
                    self.grad += safe_divide(grad.data, self.data);
                }
            }

            "tanh" => {
                self.grad += 1.0 - (self.data.tanh() * self.data.tanh());
            }

            "relu" => {
                if self.data > 0.0 {
                    self.grad = 1.0
                } else if self.data < 0.0 {
                    self.grad = 0.0
                } else {
                    self.grad = f64::NAN;
                }
            }
            _ => {}
        }
    }
}

// Formatted display output
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

// Addition support
impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value {
            data: self.data + other.data,
            grad: self.grad,
            op: "+".to_string(),
        }
    }
}

// Subtraction support
impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        Value {
            data: self.data - other.data,
            grad: self.grad,
            op: "-".to_string(),
        }
    }
}

// Multiplication support
impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        Value {
            data: self.data * other.data,
            grad: self.grad,
            op: "*".to_string(),
        }
    }
}

// Divide support
impl Div for &Value {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        Value {
            data: self.data / other.data,
            grad: self.grad,
            op: "/".to_string(),
        }
    }
}
