//! Activations as layers, thin wrappers over the [`Tensor`] methods.

use crate::tensor::Tensor;

use super::module::Module;

macro_rules! activation {
    ($(#[$doc:meta])* $name:ident => $method:ident) => {
        $(#[$doc])*
        pub struct $name;

        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.$method()
            }
        }
    };
}

activation!(/// `max(0, x)`.
            ReLU => relu);
activation!(/// `1 / (1 + e^-x)`.
            Sigmoid => sigmoid);
activation!(/// Hyperbolic tangent.
            Tanh => tanh);
activation!(/// Exact GELU, the transformer default.
            GELU => gelu);
activation!(/// SiLU / swish, `x · sigmoid(x)`.
            SiLU => silu);
activation!(/// Softmax over the last dimension.
            ///
            /// Skip it before [`cross_entropy`](super::cross_entropy), which has its own.
            Softmax => softmax);

const DEFAULT_SLOPE: f32 = 0.01;

/// ReLU with a small slope for negative inputs.
pub struct LeakyReLU {
    pub slope: f32,
}

impl LeakyReLU {
    pub fn new(slope: f32) -> LeakyReLU {
        LeakyReLU { slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        LeakyReLU {
            slope: DEFAULT_SLOPE,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.slope)
    }
}
