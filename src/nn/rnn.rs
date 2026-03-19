use crate::tensor::Tensor;
use crate::nn::module::Module;

/// Long Short-Term Memory (LSTM) layer.
pub struct LSTM {
    // Gates: input, forget, cell, output — combined into one weight matrix for efficiency
    pub weight_ih: Tensor, // [4 * hidden_size, input_size]
    pub weight_hh: Tensor, // [4 * hidden_size, hidden_size]
    pub bias_ih: Tensor,   // [4 * hidden_size]
    pub bias_hh: Tensor,   // [4 * hidden_size]
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let gate_size = 4 * hidden_size;
        let mut weight_ih = Tensor::xavier_uniform(&[gate_size, input_size], input_size, hidden_size);
        weight_ih.set_requires_grad(true);
        let mut weight_hh = Tensor::xavier_uniform(&[gate_size, hidden_size], hidden_size, hidden_size);
        weight_hh.set_requires_grad(true);
        let mut bias_ih = Tensor::zeros(&[gate_size]);
        bias_ih.set_requires_grad(true);
        let mut bias_hh = Tensor::zeros(&[gate_size]);
        bias_hh.set_requires_grad(true);

        // Initialize forget gate bias to 1.0 (helps training)
        {
            let b = bias_ih.data_mut();
            for i in hidden_size..2 * hidden_size {
                b[i] = 1.0;
            }
        }

        LSTM {
            weight_ih, weight_hh, bias_ih, bias_hh,
            input_size, hidden_size, num_layers,
        }
    }

    /// Forward pass. Returns (output, (h_n, c_n)).
    /// input: [batch, seq_len, input_size]
    /// Returns output: [batch, seq_len, hidden_size]
    pub fn forward_seq(&self, input: &Tensor, initial_state: Option<(&Tensor, &Tensor)>) -> (Tensor, Tensor, Tensor) {
        let shape = input.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        let mut h = if let Some((h0, _)) = initial_state {
            h0.clone()
        } else {
            Tensor::zeros(&[batch, self.hidden_size])
        };

        let mut c = if let Some((_, c0)) = initial_state {
            c0.clone()
        } else {
            Tensor::zeros(&[batch, self.hidden_size])
        };

        let mut outputs = Vec::with_capacity(seq_len);
        let hs = self.hidden_size;

        for t in 0..seq_len {
            // Extract input at time t: [batch, input_size]
            let x_t_data: Vec<f32> = (0..batch).flat_map(|b| {
                let offset = (b * seq_len + t) * self.input_size;
                input.to_vec()[offset..offset + self.input_size].to_vec()
            }).collect();
            let x_t = Tensor::from_vec(x_t_data, &[batch, self.input_size]);

            // gates = x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh
            let gates_i = x_t.matmul(&self.weight_ih.transpose());
            let gates_h = h.matmul(&self.weight_hh.transpose());
            let bias = self.bias_ih.reshape(&[1, 4 * hs as i64]).expand(&[batch, 4 * hs]);
            let bias2 = self.bias_hh.reshape(&[1, 4 * hs as i64]).expand(&[batch, 4 * hs]);
            let gates = gates_i.add(&gates_h).add(&bias).add(&bias2);

            let gates_data = gates.to_vec();
            let mut new_h = vec![0.0f32; batch * hs];
            let mut new_c = vec![0.0f32; batch * hs];
            let c_data = c.to_vec();

            for b in 0..batch {
                for j in 0..hs {
                    let i_gate = sigmoid(gates_data[b * 4 * hs + j]);
                    let f_gate = sigmoid(gates_data[b * 4 * hs + hs + j]);
                    let g_gate = gates_data[b * 4 * hs + 2 * hs + j].tanh();
                    let o_gate = sigmoid(gates_data[b * 4 * hs + 3 * hs + j]);

                    new_c[b * hs + j] = f_gate * c_data[b * hs + j] + i_gate * g_gate;
                    new_h[b * hs + j] = o_gate * new_c[b * hs + j].tanh();
                }
            }

            h = Tensor::from_vec(new_h, &[batch, hs]);
            c = Tensor::from_vec(new_c, &[batch, hs]);
            outputs.push(h.clone());
        }

        // Stack outputs: [batch, seq_len, hidden_size]
        let output_data: Vec<f32> = (0..batch).flat_map(|b| {
            (0..seq_len).flat_map(|t| {
                outputs[t].to_vec()[b * hs..(b + 1) * hs].to_vec()
            }).collect::<Vec<f32>>()
        }).collect();

        let output = Tensor::from_vec(output_data, &[batch, seq_len, hs]);
        (output, h, c)
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _, _) = self.forward_seq(input, None);
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight_ih.clone(), self.weight_hh.clone(), self.bias_ih.clone(), self.bias_hh.clone()]
    }
}

/// Gated Recurrent Unit (GRU) layer.
pub struct GRU {
    pub weight_ih: Tensor, // [3 * hidden_size, input_size]
    pub weight_hh: Tensor, // [3 * hidden_size, hidden_size]
    pub bias_ih: Tensor,
    pub bias_hh: Tensor,
    input_size: usize,
    hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let gate_size = 3 * hidden_size;
        let mut weight_ih = Tensor::xavier_uniform(&[gate_size, input_size], input_size, hidden_size);
        weight_ih.set_requires_grad(true);
        let mut weight_hh = Tensor::xavier_uniform(&[gate_size, hidden_size], hidden_size, hidden_size);
        weight_hh.set_requires_grad(true);
        let mut bias_ih = Tensor::zeros(&[gate_size]);
        bias_ih.set_requires_grad(true);
        let mut bias_hh = Tensor::zeros(&[gate_size]);
        bias_hh.set_requires_grad(true);

        GRU { weight_ih, weight_hh, bias_ih, bias_hh, input_size, hidden_size }
    }

    pub fn forward_seq(&self, input: &Tensor, h0: Option<&Tensor>) -> (Tensor, Tensor) {
        let shape = input.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);
        let hs = self.hidden_size;

        let mut h = h0.cloned().unwrap_or_else(|| Tensor::zeros(&[batch, hs]));
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t_data: Vec<f32> = (0..batch).flat_map(|b| {
                let offset = (b * seq_len + t) * self.input_size;
                input.to_vec()[offset..offset + self.input_size].to_vec()
            }).collect();
            let x_t = Tensor::from_vec(x_t_data, &[batch, self.input_size]);

            let gates_i = x_t.matmul(&self.weight_ih.transpose());
            let gates_h = h.matmul(&self.weight_hh.transpose());
            let bi = self.bias_ih.reshape(&[1, 3 * hs as i64]).expand(&[batch, 3 * hs]);
            let bh = self.bias_hh.reshape(&[1, 3 * hs as i64]).expand(&[batch, 3 * hs]);

            let gi = gates_i.add(&bi).to_vec();
            let gh = gates_h.add(&bh).to_vec();
            let h_data = h.to_vec();

            let mut new_h = vec![0.0f32; batch * hs];
            for b in 0..batch {
                for j in 0..hs {
                    let r = sigmoid(gi[b * 3 * hs + j] + gh[b * 3 * hs + j]);
                    let z = sigmoid(gi[b * 3 * hs + hs + j] + gh[b * 3 * hs + hs + j]);
                    let n = (gi[b * 3 * hs + 2 * hs + j] + r * gh[b * 3 * hs + 2 * hs + j]).tanh();
                    new_h[b * hs + j] = (1.0 - z) * n + z * h_data[b * hs + j];
                }
            }

            h = Tensor::from_vec(new_h, &[batch, hs]);
            outputs.push(h.clone());
        }

        let output_data: Vec<f32> = (0..batch).flat_map(|b| {
            (0..seq_len).flat_map(|t| {
                outputs[t].to_vec()[b * hs..(b + 1) * hs].to_vec()
            }).collect::<Vec<f32>>()
        }).collect();

        let output = Tensor::from_vec(output_data, &[batch, seq_len, hs]);
        (output, h)
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let (output, _) = self.forward_seq(input, None);
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight_ih.clone(), self.weight_hh.clone(), self.bias_ih.clone(), self.bias_hh.clone()]
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
