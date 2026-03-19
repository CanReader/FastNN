use crate::tensor::Tensor;
use crate::nn::module::Module;

pub struct ReLU;
impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor { input.relu() }
}

pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor { input.sigmoid() }
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor { input.tanh_act() }
}

pub struct GELU;
impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor { input.gelu() }
}

pub struct SiLU;
impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Tensor { input.silu() }
}

pub struct LeakyReLU {
    pub negative_slope: f32,
}
impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self { LeakyReLU { negative_slope } }
}
impl Default for LeakyReLU {
    fn default() -> Self { LeakyReLU { negative_slope: 0.01 } }
}
impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor { input.leaky_relu(self.negative_slope) }
}

pub struct Softmax;
impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor { input.softmax() }
}
