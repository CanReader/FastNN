use crate::optim::Optimizer;

/// Learning rate scheduler trait.
pub trait LRScheduler {
    /// Update the learning rate based on the current epoch/step.
    fn step(&mut self, optimizer: &mut dyn Optimizer);

    /// Get the current scheduled learning rate.
    fn get_lr(&self) -> f32;
}

/// Step decay: multiply LR by gamma every `step_size` epochs.
pub struct StepLR {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR { base_lr, step_size, gamma, current_epoch: 0, current_lr: base_lr }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_epoch += 1;
        if self.current_epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f32 { self.current_lr }
}

/// Cosine annealing with optional minimum LR.
pub struct CosineAnnealingLR {
    base_lr: f32,
    min_lr: f32,
    total_steps: usize,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, total_steps: usize) -> Self {
        CosineAnnealingLR { base_lr, min_lr: 0.0, total_steps, current_step: 0 }
    }

    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let progress = self.current_step as f32 / self.total_steps as f32;
        let lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos());
        optimizer.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        let progress = self.current_step as f32 / self.total_steps as f32;
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Linear warmup for the first N steps.
pub struct LinearWarmup {
    base_lr: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl LinearWarmup {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        LinearWarmup { base_lr, warmup_steps, current_step: 0 }
    }
}

impl LRScheduler for LinearWarmup {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        if self.current_step <= self.warmup_steps {
            let lr = self.base_lr * self.current_step as f32 / self.warmup_steps as f32;
            optimizer.set_lr(lr);
        }
    }

    fn get_lr(&self) -> f32 {
        if self.current_step <= self.warmup_steps {
            self.base_lr * self.current_step as f32 / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }
}

/// One-cycle learning rate policy (warmup + cosine decay).
pub struct OneCycleLR {
    max_lr: f32,
    total_steps: usize,
    pct_start: f32,
    div_factor: f32,
    final_div_factor: f32,
    current_step: usize,
}

impl OneCycleLR {
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        OneCycleLR {
            max_lr,
            total_steps,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 10000.0,
            current_step: 0,
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        let warmup_steps = (self.pct_start * self.total_steps as f32) as usize;
        let initial_lr = self.max_lr / self.div_factor;
        let min_lr = self.max_lr / self.final_div_factor;

        if self.current_step <= warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / warmup_steps as f32;
            initial_lr + (self.max_lr - initial_lr) * progress
        } else {
            // Cosine decay
            let progress = (self.current_step - warmup_steps) as f32 / (self.total_steps - warmup_steps) as f32;
            min_lr + 0.5 * (self.max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}
