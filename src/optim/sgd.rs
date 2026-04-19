use crate::tensor::Tensor;
use crate::optim::Optimizer;

/// Stochastic Gradient Descent with optional momentum and weight decay.
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocities: Vec::new(),
        }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        if self.velocities.len() != params.len() {
            self.velocities = vec![None; params.len()];
        }

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue, // no grad recorded — skip (e.g. unused parameter)
            };

            let mut g = grad.clone();

            // L2 regularization
            if self.weight_decay != 0.0 {
                g = g.add(&param.mul_scalar(self.weight_decay));
            }

            let update = if self.momentum != 0.0 {
                let v = match &self.velocities[i] {
                    Some(v) => v.mul_scalar(self.momentum).add(&g.mul_scalar(1.0 - self.dampening)),
                    None => g.clone(),
                };
                let update = if self.nesterov {
                    g.add(&v.mul_scalar(self.momentum))
                } else {
                    v.clone()
                };
                self.velocities[i] = Some(v);
                update
            } else {
                g
            };

            param.apply_sgd_update(self.lr, &update);
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
