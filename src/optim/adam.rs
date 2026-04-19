use crate::tensor::Tensor;
use crate::optim::Optimizer;

/// Adam optimizer.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step_count: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    initialized: bool,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        if !self.initialized {
            self.m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
            self.v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
            self.initialized = true;
        }

        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let mut param_data = param.to_vec();
            let grad_data = grad.to_vec();
            let m_data = &mut self.m[i];
            let v_data = &mut self.v[i];

            for j in 0..param_data.len() {
                let g = if self.weight_decay != 0.0 {
                    grad_data[j] + self.weight_decay * param_data[j]
                } else {
                    grad_data[j]
                };
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m_data[j] / bc1;
                let v_hat = v_data[j] / bc2;
                param_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }

            param.set_data_from_vec(param_data);
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}

/// AdamW optimizer (decoupled weight decay).
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step_count: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    initialized: bool,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        if !self.initialized {
            self.m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
            self.v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
            self.initialized = true;
        }

        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let mut param_data = param.to_vec();
            let grad_data = grad.to_vec();
            let m_data = &mut self.m[i];
            let v_data = &mut self.v[i];

            for j in 0..param_data.len() {
                // Decoupled weight decay applied directly to params.
                param_data[j] -= self.lr * self.weight_decay * param_data[j];

                let g = grad_data[j];
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m_data[j] / bc1;
                let v_hat = v_data[j] / bc2;
                param_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }

            param.set_data_from_vec(param_data);
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
