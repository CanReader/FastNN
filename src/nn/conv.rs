use std::collections::HashMap;
use std::sync::Arc;
use crate::tensor::Tensor;
use crate::nn::module::Module;
use crate::autograd::graph;
use crate::autograd::backward_ops::Conv2dBackward;

/// 2D Convolution layer.
pub struct Conv2d {
    pub weight: Tensor, // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self::with_params(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), (padding, padding), true)
    }

    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let mut weight = Tensor::kaiming_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            fan_in,
        );
        weight.set_requires_grad(true);

        let bias = if use_bias {
            let bound = 1.0 / (fan_in as f32).sqrt();
            let mut b = Tensor::from_vec(
                (0..out_channels).map(|_| {
                    use rand::Rng;
                    rand::thread_rng().gen_range(-bound..bound)
                }).collect(),
                &[out_channels],
            );
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Output spatial dimensions for a given input size.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (input_w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_channels, H, W]
        let shape = input.shape();
        assert_eq!(shape.len(), 4, "Conv2d expects 4D input [N, C, H, W]");
        let (batch_size, in_c, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(in_c, self.in_channels);

        let (out_h, out_w) = self.output_size(in_h, in_w);
        let input_data = input.to_vec();
        let weight_data = self.weight.to_vec();

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let mut output = vec![0.0f32; batch_size * self.out_channels * out_h * out_w];

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = if let Some(ref bias) = self.bias {
                            bias.data()[oc]
                        } else {
                            0.0f32
                        };
                        for ic in 0..self.in_channels {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = (oh * sh + ki) as isize - ph as isize;
                                    let iw = (ow * sw + kj) as isize - pw as isize;
                                    if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                        let input_idx = ((b * in_c + ic) * in_h + ih as usize) * in_w + iw as usize;
                                        let weight_idx = ((oc * self.in_channels + ic) * kh + ki) * kw + kj;
                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                        output[((b * self.out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                    }
                }
            }
        }

        let mut out = Tensor::from_vec(output, &[batch_size, self.out_channels, out_h, out_w]);

        let any_requires_grad = input.requires_grad()
            || self.weight.requires_grad()
            || self.bias.as_ref().map_or(false, |b| b.requires_grad());

        if graph::is_grad_enabled() && any_requires_grad {
            out.set_requires_grad(true);

            // input_ids and leaf cells: always [input, weight, bias?]
            let has_bias = self.bias.is_some();
            let mut input_ids = vec![input.id(), self.weight.id()];
            let mut leaf_cells = Vec::new();
            if input.requires_grad() {
                leaf_cells.push((input.id(), input.grad_cell()));
            }
            if self.weight.requires_grad() {
                leaf_cells.push((self.weight.id(), self.weight.grad_cell()));
            }
            if let Some(ref bias) = self.bias {
                input_ids.push(bias.id());
                if bias.requires_grad() {
                    leaf_cells.push((bias.id(), bias.grad_cell()));
                }
            }

            let grad_fn = Arc::new(Conv2dBackward {
                input_ids,
                input: input.clone(),
                weight: self.weight.clone(),
                has_bias,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
            });

            graph::record_op_with_cells(grad_fn, out.id(), leaf_cells);
        }

        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }
}
