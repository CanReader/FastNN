use crate::tensor::Tensor;
use crate::nn::module::Module;

/// Dropout regularization layer.
pub struct Dropout {
    pub p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        use rand::Rng;
        let data = input.to_vec();
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);
        let result: Vec<f32> = data.iter().map(|&x| {
            if rng.gen::<f32>() > self.p { x * scale } else { 0.0 }
        }).collect();

        Tensor::from_vec(result, input.shape()).to_device(input.device())
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
}
