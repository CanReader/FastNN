use crate::tensor::Tensor;

/// Cross-entropy loss for classification.
/// Expects logits [batch, num_classes] and targets [batch] (class indices).
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self { CrossEntropyLoss }

    pub fn forward(&self, logits: &Tensor, targets: &[usize]) -> Tensor {
        let shape = logits.shape();
        assert_eq!(shape.len(), 2, "Expected [batch, num_classes]");
        let (batch_size, num_classes) = (shape[0], shape[1]);
        assert_eq!(targets.len(), batch_size);

        let log_probs = logits.log_softmax();
        let log_prob_data = log_probs.to_vec();

        let mut loss = 0.0f32;
        for (b, &target) in targets.iter().enumerate() {
            assert!(target < num_classes, "Target {} out of range", target);
            loss -= log_prob_data[b * num_classes + target];
        }
        loss /= batch_size as f32;

        Tensor::from_vec(vec![loss], &[1])
    }

    /// Forward returning both loss and gradient w.r.t. logits.
    pub fn forward_with_grad(&self, logits: &Tensor, targets: &[usize]) -> (Tensor, Tensor) {
        let shape = logits.shape();
        let (batch_size, num_classes) = (shape[0], shape[1]);

        let probs = logits.softmax();
        let prob_data = probs.to_vec();
        let log_probs = logits.log_softmax();
        let log_prob_data = log_probs.to_vec();

        let mut loss = 0.0f32;
        let mut grad = vec![0.0f32; batch_size * num_classes];

        for (b, &target) in targets.iter().enumerate() {
            loss -= log_prob_data[b * num_classes + target];
            for c in 0..num_classes {
                grad[b * num_classes + c] = (prob_data[b * num_classes + c] - if c == target { 1.0 } else { 0.0 }) / batch_size as f32;
            }
        }
        loss /= batch_size as f32;

        (Tensor::from_vec(vec![loss], &[1]), Tensor::from_vec(grad, &[batch_size, num_classes]))
    }
}

/// Mean Squared Error loss.
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self { MSELoss }

    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let diff = predictions.sub(targets);
        let sq = diff.mul(&diff);
        sq.mean()
    }

    pub fn forward_with_grad(&self, predictions: &Tensor, targets: &Tensor) -> (Tensor, Tensor) {
        let diff = predictions.sub(targets);
        let sq = diff.mul(&diff);
        let loss = sq.mean();
        let n = predictions.numel() as f32;
        let grad = diff.mul_scalar(2.0 / n);
        (loss, grad)
    }
}

/// Binary Cross-Entropy loss (expects probabilities in [0, 1]).
pub struct BCELoss;

impl BCELoss {
    pub fn new() -> Self { BCELoss }

    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let eps = 1e-7;
        let pred = predictions.clamp(eps, 1.0 - eps);
        let log_p = pred.log();
        let one = Tensor::ones(predictions.shape());
        let log_1mp = one.sub(&pred).log();

        // loss = -mean(t * log(p) + (1-t) * log(1-p))
        let term1 = targets.mul(&log_p);
        let term2 = one.sub(targets).mul(&log_1mp);
        term1.add(&term2).neg().mean()
    }
}

/// BCE with logits (numerically stable — applies sigmoid internally).
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self { BCEWithLogitsLoss }

    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        // Numerically stable: max(x, 0) - x*t + log(1 + exp(-abs(x)))
        let data = logits.to_vec();
        let target_data = targets.to_vec();
        let n = data.len() as f32;

        let loss: f32 = data.iter().zip(target_data.iter()).map(|(&x, &t)| {
            let max_val = x.max(0.0);
            max_val - x * t + ((-max_val).exp() + (x - max_val).exp()).ln()
        }).sum::<f32>() / n;

        Tensor::from_vec(vec![loss], &[1])
    }

    pub fn forward_with_grad(&self, logits: &Tensor, targets: &Tensor) -> (Tensor, Tensor) {
        let loss = self.forward(logits, targets);
        let probs = logits.sigmoid();
        let n = logits.numel() as f32;
        let grad = probs.sub(targets).mul_scalar(1.0 / n);
        (loss, grad)
    }
}
