//! The safetensors interchange format — how weights arrive from elsewhere.
//!
//! The format is deliberately simple: an 8-byte little-endian header length, a
//! JSON header mapping each tensor name to `{dtype, shape, data_offsets}`, and
//! the raw tensor bytes. No code execution, no pickle — which is exactly why
//! the wider ecosystem standardized on it for distributing weights.
//!
//! ```text
//! u64 header_len │ {"w": {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}, …} │ data…
//! ```
//!
//! Writing always stores `F32`, FastNN's compute type. Reading additionally
//! accepts `F16` and `BF16`, widening on load — exact, since every 16-bit
//! float is representable in `f32`. Loading is *partial by design*: what
//! matches is loaded, and the [`LoadReport`] names what was missing and what
//! went unused, because a checkpoint from another framework rarely aligns
//! perfectly on the first try and "which keys?" is the whole debugging session.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::Tensor;

use super::half::{f32_from_bf16, f32_from_f16};

/// What a partial load actually did. Empty `missing` and `unexpected` mean the
/// file and the model agreed exactly.
#[derive(Debug, Default)]
pub struct LoadReport {
    /// Parameters and buffers that received values.
    pub loaded: Vec<String>,
    /// Model tensors the file had nothing for — they keep their current values.
    pub missing: Vec<String>,
    /// File tensors no model tensor claimed.
    pub unexpected: Vec<String>,
}

/// Write a module's parameters and buffers as `F32` safetensors.
pub fn save_safetensors(module: &dyn Module, path: impl AsRef<Path>) -> Result<()> {
    // BTreeMap for a stable on-disk order, as the .fdl writer does.
    let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, param) in module.named_parameters() {
        tensors.insert(name, param.value().cpu());
    }
    for (name, buffer) in module.named_buffers() {
        tensors.insert(name, buffer.value().cpu());
    }

    let mut header = String::from("{");
    let mut offset = 0usize;
    for (index, (name, tensor)) in tensors.iter().enumerate() {
        let bytes = tensor.numel() * 4;
        if index > 0 {
            header.push(',');
        }
        let dims: Vec<String> = tensor.shape().iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            escape(name),
            dims.join(","),
            offset,
            offset + bytes
        ));
        offset += bytes;
    }
    header.push('}');

    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for tensor in tensors.values() {
        for value in tensor.to_vec() {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Load matching tensors from a safetensors file into a module.
///
/// Shapes must agree wherever a name matches — that is an error, not a report
/// entry, because silently skipping a shape clash loads a model that *looks*
/// initialized. Names that simply do not match are reported and skipped.
pub fn load_safetensors(module: &dyn Module, path: impl AsRef<Path>) -> Result<LoadReport> {
    load_safetensors_renamed(module, path, &HashMap::new())
}

/// [`load_safetensors`] through a name map: `file key → FastNN parameter path`.
///
/// The declarative escape hatch for foreign checkpoints — import logic becomes
/// a table (`"transformer.h.0.attn.c_attn.weight"` → `"blocks.0.…"`), not code.
pub fn load_safetensors_renamed(
    module: &dyn Module,
    path: impl AsRef<Path>,
    rename: &HashMap<String, String>,
) -> Result<LoadReport> {
    let bytes = std::fs::read(path)?;
    let (entries, data) = parse_file(&bytes)?;

    // Index by the *effective* name: the mapping applied to the file's keys.
    let mut by_name: HashMap<&str, &Entry> = HashMap::new();
    for entry in &entries {
        let effective = rename
            .get(&entry.name)
            .map_or(entry.name.as_str(), String::as_str);
        by_name.insert(effective, entry);
    }

    let mut report = LoadReport::default();
    let mut used: Vec<&str> = Vec::new();

    let mut fill = |name: String, shape: Vec<usize>, place: &mut dyn FnMut(Tensor)| -> Result<()> {
        let Some(entry) = by_name.get(name.as_str()) else {
            report.missing.push(name);
            return Ok(());
        };
        if entry.shape != shape {
            return Err(Error::Checkpoint(format!(
                "safetensors: '{name}' is {:?} in the file but {shape:?} in the model",
                entry.shape
            )));
        }
        place(Tensor::from_vec(entry.decode(data)?, &shape));
        used.push(by_name.get_key_value(name.as_str()).unwrap().0);
        report.loaded.push(name);
        Ok(())
    };

    for (name, param) in module.named_parameters() {
        let shape = param.shape();
        fill(name, shape, &mut |tensor| {
            param.set_value(tensor.to(param.device()))
        })?;
    }
    for (name, buffer) in module.named_buffers() {
        let device = buffer.value().device();
        let shape = buffer.value().shape().to_vec();
        fill(name, shape, &mut |tensor| {
            buffer.set_value(tensor.to(device))
        })?;
    }

    for entry in &entries {
        let effective = rename
            .get(&entry.name)
            .map_or(entry.name.as_str(), String::as_str);
        if !used.contains(&effective) {
            report.unexpected.push(entry.name.clone());
        }
    }
    Ok(report)
}

// ── The header ───────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn bytes(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
        }
    }
}

#[derive(Debug)]
struct Entry {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    offsets: (usize, usize),
}

impl Entry {
    /// The tensor's bytes, widened to `f32`.
    fn decode(&self, data: &[u8]) -> Result<Vec<f32>> {
        let bytes = &data[self.offsets.0..self.offsets.1];
        Ok(match self.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| f32_from_f16(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32_from_bf16(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        })
    }
}

fn bad(message: impl Into<String>) -> Error {
    Error::Checkpoint(format!("safetensors: {}", message.into()))
}

/// Split a file into validated header entries and the data section.
fn parse_file(bytes: &[u8]) -> Result<(Vec<Entry>, &[u8])> {
    if bytes.len() < 8 {
        return Err(bad("file shorter than its header length field"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let Some(data_start) = header_len.checked_add(8).filter(|&end| end <= bytes.len()) else {
        return Err(bad(format!("header length {header_len} exceeds the file")));
    };
    let header =
        std::str::from_utf8(&bytes[8..data_start]).map_err(|_| bad("header is not valid UTF-8"))?;
    let data = &bytes[data_start..];

    let entries = Header::new(header).parse()?;
    for entry in &entries {
        let (start, end) = entry.offsets;
        let numel: usize = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad(format!("'{}' has an overflowing shape", entry.name)))?;
        if start > end || end > data.len() {
            return Err(bad(format!(
                "'{}' points outside the data section",
                entry.name
            )));
        }
        if end - start != numel * entry.dtype.bytes() {
            return Err(bad(format!(
                "'{}' claims {:?} ({numel} values) but spans {} bytes",
                entry.name,
                entry.shape,
                end - start
            )));
        }
    }
    Ok((entries, data))
}

/// A hand-rolled parser for exactly the JSON the format specifies: one flat
/// object of tensor entries plus an optional string-to-string `__metadata__`.
///
/// Small enough to read in one sitting on purpose — the alternative is a JSON
/// dependency for one fixed schema.
struct Header<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Header<'a> {
    fn new(text: &'a str) -> Header<'a> {
        Header {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn parse(mut self) -> Result<Vec<Entry>> {
        let mut entries = Vec::new();
        self.expect(b'{')?;
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(entries);
        }
        loop {
            let name = self.string()?;
            self.expect(b':')?;
            if name == "__metadata__" {
                self.skip_string_object()?;
            } else {
                entries.push(self.entry(name)?);
            }
            match self.next()? {
                b',' => continue,
                b'}' => return Ok(entries),
                other => {
                    return Err(bad(format!(
                        "expected ',' or '}}', found '{}'",
                        other as char
                    )))
                }
            }
        }
    }

    /// `{"dtype": "...", "shape": [...], "data_offsets": [a, b]}`, keys in any order.
    fn entry(&mut self, name: String) -> Result<Entry> {
        let (mut dtype, mut shape, mut offsets) = (None, None, None);
        self.expect(b'{')?;
        loop {
            let key = self.string()?;
            self.expect(b':')?;
            match key.as_str() {
                "dtype" => {
                    dtype = Some(match self.string()?.as_str() {
                        "F32" => Dtype::F32,
                        "F16" => Dtype::F16,
                        "BF16" => Dtype::Bf16,
                        other => {
                            return Err(bad(format!(
                        "'{name}' has dtype {other}, and only F32/F16/BF16 load into f32 storage"
                    )))
                        }
                    })
                }
                "shape" => shape = Some(self.numbers()?),
                "data_offsets" => {
                    let pair = self.numbers()?;
                    if pair.len() != 2 {
                        return Err(bad(format!("'{name}' offsets need exactly [start, end]")));
                    }
                    offsets = Some((pair[0], pair[1]));
                }
                other => return Err(bad(format!("unknown key '{other}' in '{name}'"))),
            }
            match self.next()? {
                b',' => continue,
                b'}' => break,
                other => {
                    return Err(bad(format!(
                        "expected ',' or '}}', found '{}'",
                        other as char
                    )))
                }
            }
        }
        match (dtype, shape, offsets) {
            (Some(dtype), Some(shape), Some(offsets)) => Ok(Entry {
                name,
                dtype,
                shape,
                offsets,
            }),
            _ => Err(bad(format!(
                "'{name}' is missing dtype, shape, or data_offsets"
            ))),
        }
    }

    /// `[1, 2, 3]` (or `[]`), as machine sizes.
    fn numbers(&mut self) -> Result<Vec<usize>> {
        let mut values = Vec::new();
        self.expect(b'[')?;
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(values);
        }
        loop {
            values.push(self.number()?);
            match self.next()? {
                b',' => continue,
                b']' => return Ok(values),
                other => {
                    return Err(bad(format!(
                        "expected ',' or ']', found '{}'",
                        other as char
                    )))
                }
            }
        }
    }

    fn number(&mut self) -> Result<usize> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(bad("expected a number"));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap()
            .parse()
            .map_err(|_| bad("number too large"))
    }

    /// A JSON string with the standard escapes, including `\uXXXX` pairs.
    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.raw_byte()? {
                b'"' => return Ok(out),
                b'\\' => match self.raw_byte()? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => out.push(self.unicode_escape()?),
                    other => return Err(bad(format!("unknown escape '\\{}'", other as char))),
                },
                byte if byte < 0x20 => return Err(bad("raw control character in string")),
                byte => {
                    // Re-assemble multi-byte UTF-8 (the header is validated UTF-8).
                    let width = match byte {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    let start = self.pos - 1;
                    self.pos = start + width;
                    out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).unwrap());
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char> {
        let unit = self.hex4()?;
        // Surrogate pair: a high half must be followed by an escaped low half.
        if (0xD800..0xDC00).contains(&unit) {
            if self.raw_byte()? != b'\\' || self.raw_byte()? != b'u' {
                return Err(bad("lone high surrogate"));
            }
            let low = self.hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return Err(bad("invalid low surrogate"));
            }
            let code = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
            return char::from_u32(code).ok_or_else(|| bad("invalid surrogate pair"));
        }
        char::from_u32(unit).ok_or_else(|| bad("invalid unicode escape"))
    }

    fn hex4(&mut self) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let digit = (self.raw_byte()? as char)
                .to_digit(16)
                .ok_or_else(|| bad("invalid hex in unicode escape"))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    /// `{"key": "value", ...}` — the metadata object, values discarded.
    fn skip_string_object(&mut self) -> Result<()> {
        self.expect(b'{')?;
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.string()?;
            self.expect(b':')?;
            self.string()?;
            match self.next()? {
                b',' => continue,
                b'}' => return Ok(()),
                other => {
                    return Err(bad(format!(
                        "expected ',' or '}}', found '{}'",
                        other as char
                    )))
                }
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Result<u8> {
        self.skip_whitespace();
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| bad("header ended early"))
    }

    fn next(&mut self) -> Result<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Ok(byte)
    }

    /// The next byte with no whitespace skipping — inside strings, spaces count.
    fn raw_byte(&mut self) -> Result<u8> {
        let byte = self
            .bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| bad("header ended early"))?;
        self.pos += 1;
        Ok(byte)
    }

    fn expect(&mut self, wanted: u8) -> Result<()> {
        let got = self.next()?;
        if got != wanted {
            return Err(bad(format!(
                "expected '{}', found '{}'",
                wanted as char, got as char
            )));
        }
        Ok(())
    }
}

/// Escape a tensor name for the JSON header.
fn escape(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
