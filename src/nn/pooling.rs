//! Pooling layers over `[N, C, H, W]`.

use crate::tensor::{Tensor, Window};

use super::module::Module;

/// Keep the largest value in each window.
pub struct MaxPool2d {
    window: Window,
}

impl MaxPool2d {
    /// A `kernel × kernel` window that does not overlap — the usual downsampler.
    pub fn new(kernel: usize) -> MaxPool2d {
        MaxPool2d {
            window: Window::square(kernel, kernel, 0),
        }
    }

    pub fn with_window(window: Window) -> MaxPool2d {
        MaxPool2d { window }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(self.window)
    }
}

/// Average the values in each window.
pub struct AvgPool2d {
    window: Window,
}

impl AvgPool2d {
    pub fn new(kernel: usize) -> AvgPool2d {
        AvgPool2d {
            window: Window::square(kernel, kernel, 0),
        }
    }

    pub fn with_window(window: Window) -> AvgPool2d {
        AvgPool2d { window }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.avg_pool2d(self.window)
    }
}

/// Average into a fixed output size, whatever the input resolution.
pub struct AdaptiveAvgPool2d {
    output: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output: (usize, usize)) -> AdaptiveAvgPool2d {
        AdaptiveAvgPool2d { output }
    }

    /// Collapse each channel to a single number.
    pub fn global() -> AdaptiveAvgPool2d {
        AdaptiveAvgPool2d { output: (1, 1) }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_avg_pool2d(self.output)
    }
}
