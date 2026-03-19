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
    m: Vec<Tensor>,  // first moment
    v: Vec<Tensor>,  // second moment
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
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(params.len(), grads.len());

        if !self.initialized {
            self.m = params.iter().map(|p| Tensor::zeros(p.shape())).collect();
            self.v = params.iter().map(|p| Tensor::zeros(p.shape())).collect();
            self.initialized = true;
        }

        self.step_count += 1;
        let t = self.step_count as f32;

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut param_data = param.to_vec();
            let grad_data = grad.to_vec();
            let mut m_data = self.m[i].to_vec();
            let mut v_data = self.v[i].to_vec();

            let bias_correction1 = 1.0 - self.beta1.powf(t);
            let bias_correction2 = 1.0 - self.beta2.powf(t);

            for j in 0..param_data.len() {
                let g = grad_data[j];

                // L2 regularization (classic Adam, not decoupled)
                let g = if self.weight_decay != 0.0 {
                    g + self.weight_decay * param_data[j]
                } else {
                    g
                };

                // Update moments
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;

                // Bias-corrected moments
                let m_hat = m_data[j] / bias_correction1;
                let v_hat = v_data[j] / bias_correction2;

                param_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }

            self.m[i] = Tensor::from_vec(m_data, param.shape());
            self.v[i] = Tensor::from_vec(v_data, param.shape());
            *param = Tensor::from_vec(param_data, param.shape());
            param.set_requires_grad(true);
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
    m: Vec<Tensor>,
    v: Vec<Tensor>,
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
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        assert_eq!(params.len(), grads.len());

        if !self.initialized {
            self.m = params.iter().map(|p| Tensor::zeros(p.shape())).collect();
            self.v = params.iter().map(|p| Tensor::zeros(p.shape())).collect();
            self.initialized = true;
        }

        self.step_count += 1;
        let t = self.step_count as f32;

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut param_data = param.to_vec();
            let grad_data = grad.to_vec();
            let mut m_data = self.m[i].to_vec();
            let mut v_data = self.v[i].to_vec();

            let bias_correction1 = 1.0 - self.beta1.powf(t);
            let bias_correction2 = 1.0 - self.beta2.powf(t);

            for j in 0..param_data.len() {
                // Decoupled weight decay — applied directly to params
                param_data[j] -= self.lr * self.weight_decay * param_data[j];

                let g = grad_data[j];
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;

                let m_hat = m_data[j] / bias_correction1;
                let v_hat = v_data[j] / bias_correction2;

                param_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }

            self.m[i] = Tensor::from_vec(m_data, param.shape());
            self.v[i] = Tensor::from_vec(v_data, param.shape());
            *param = Tensor::from_vec(param_data, param.shape());
            param.set_requires_grad(true);
        }
    }

    fn get_lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
