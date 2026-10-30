//! Checkpoints: `.fdl` weights, `.fdl` training state, safetensors, and the
//! 16-bit float conversions safetensors needs.
//!
//! - [`save`] and [`load`] move a model's weights to and from a file. Shapes must
//!   match, and a tensor that moved is an error, not a silent skip.
//! - [`save_training`] and [`load_training`] add the optimizer state and the step,
//!   which is what makes a run resumable.
//! - [`save_safetensors`] and [`load_safetensors`] speak the format the rest of
//!   the ecosystem uses. F16 and BF16 load and are widened to `f32`, and
//!   [`load_safetensors_renamed`] maps foreign names onto yours and reports what
//!   matched.
//!
//! ```
//! # use fastnn::prelude::*;
//! # let path = std::env::temp_dir().join("fastnn-doc-save.fdl");
//! let model = Sequential::new().add(Linear::new(4, 2));
//! save(&model, &path)?;
//!
//! let fresh = Sequential::new().add(Linear::new(4, 2));
//! load(&fresh, &path)?;
//! assert_eq!(
//!     fresh.parameters()[0].value().to_vec(),
//!     model.parameters()[0].value().to_vec(),
//! );
//! # Ok::<(), fastnn::Error>(())
//! ```

pub mod checkpoint;
pub mod half;
pub mod safetensors;
pub mod training;

pub use checkpoint::{Checkpoint, load, load_tensors, save, save_tensors};
pub use safetensors::{LoadReport, load_safetensors, load_safetensors_renamed, save_safetensors};
pub use training::{load_training, save_training};
