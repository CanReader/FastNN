//! The safetensors reader and writer, end to end.

use std::collections::HashMap;

use fastnn::prelude::*;
use fastnn::serialize::half::{bf16_from_f32, f16_from_f32};
use fastnn::serialize::load_safetensors_renamed;

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn roundtrip_reproduces_every_prediction() {
    manual_seed(20);
    let model = Sequential::new()
        .add(Linear::new(5, 8))
        .add(GELU)
        .add(Linear::new(8, 2));
    let inputs = Tensor::randn(&[3, 5]);
    let before = no_grad(|| model.forward(&inputs)).to_vec();

    let path = temp("fastnn_st_roundtrip.safetensors");
    save_safetensors(&model, &path).unwrap();

    let reloaded = Sequential::new()
        .add(Linear::new(5, 8))
        .add(GELU)
        .add(Linear::new(8, 2));
    let report = load_safetensors(&reloaded, &path).unwrap();
    std::fs::remove_file(&path).ok();

    assert!(report.missing.is_empty(), "missing: {:?}", report.missing);
    assert!(
        report.unexpected.is_empty(),
        "unexpected: {:?}",
        report.unexpected
    );
    assert_eq!(report.loaded.len(), 4, "two layers × weight+bias");
    assert_eq!(no_grad(|| reloaded.forward(&inputs)).to_vec(), before);
}

#[test]
fn partial_load_reports_instead_of_guessing() {
    manual_seed(21);
    let small = Sequential::new().add(Linear::new(4, 4));
    let path = temp("fastnn_st_partial.safetensors");
    save_safetensors(&small, &path).unwrap();

    // A model with an extra layer: the first layer matches, the second cannot.
    let bigger = Sequential::new()
        .add(Linear::new(4, 4))
        .add(Linear::new(4, 3));
    let report = load_safetensors(&bigger, &path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(report.loaded, vec!["0.weight", "0.bias"]);
    assert_eq!(report.missing, vec!["1.weight", "1.bias"]);
    assert!(report.unexpected.is_empty());
}

#[test]
fn renaming_maps_foreign_keys_onto_the_model() {
    manual_seed(22);
    let source = Linear::new(3, 2);
    let path = temp("fastnn_st_rename.safetensors");
    save_safetensors(&source, &path).unwrap();

    // A container gives the same layer different names; the map bridges them.
    let target = Sequential::new().add(Linear::new(3, 2));
    let rename: HashMap<String, String> = [
        ("weight".to_string(), "0.weight".to_string()),
        ("bias".to_string(), "0.bias".to_string()),
    ]
    .into();
    let report = load_safetensors_renamed(&target, &path, &rename).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(report.loaded.len(), 2);
    assert!(report.missing.is_empty() && report.unexpected.is_empty());
    assert_eq!(
        target.parameters()[0].value().to_vec(),
        source.parameters()[0].value().to_vec()
    );
}

#[test]
fn shape_clash_is_an_error_that_names_the_tensor() {
    manual_seed(23);
    let source = Linear::new(3, 2);
    let path = temp("fastnn_st_clash.safetensors");
    save_safetensors(&source, &path).unwrap();

    let wrong = Linear::new(3, 5);
    let error = load_safetensors(&wrong, &path).unwrap_err().to_string();
    std::fs::remove_file(&path).ok();

    assert!(
        error.contains("weight"),
        "error should name the tensor: {error}"
    );
    assert!(
        error.contains("[2, 3]") && error.contains("[5, 3]"),
        "and both shapes: {error}"
    );
}

/// A file written by hand in the two half-precision dtypes: loading must widen
/// each stored value to exactly the f32 the conversion tables promise.
#[test]
fn half_precision_tensors_widen_exactly() {
    let f16_values = [1.0f32, -2.5, 0.15625];
    let bf16_values = [3.0f32];

    let mut data = Vec::new();
    for &v in &f16_values {
        data.extend_from_slice(&f16_from_f32(v).to_le_bytes());
    }
    for &v in &bf16_values {
        data.extend_from_slice(&bf16_from_f32(v).to_le_bytes());
    }

    let header = concat!(
        r#"{"weight":{"dtype":"F16","shape":[1,3],"data_offsets":[0,6]},"#,
        r#""bias":{"dtype":"BF16","shape":[1],"data_offsets":[6,8]},"#,
        r#""__metadata__":{"format":"pt"}}"#
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);

    let path = temp("fastnn_st_half.safetensors");
    std::fs::write(&path, file).unwrap();

    let model = Linear::new(3, 1);
    let report = load_safetensors(&model, &path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(report.loaded.len(), 2);
    let params = model.parameters();
    assert_eq!(
        params[0].value().to_vec(),
        f16_values,
        "f16 weight widened exactly"
    );
    assert_eq!(
        params[1].value().to_vec(),
        bf16_values,
        "bf16 bias widened exactly"
    );
}

#[test]
fn corrupt_headers_fail_loudly_not_quietly() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncated", vec![1, 2, 3]),
        ("header past eof", {
            let mut f = 1000u64.to_le_bytes().to_vec();
            f.extend_from_slice(b"{}");
            f
        }),
        ("offsets past data", {
            let header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
            let mut f = (header.len() as u64).to_le_bytes().to_vec();
            f.extend_from_slice(header.as_bytes());
            f.extend_from_slice(&[0u8; 4]); // only 4 of the claimed 16 bytes
            f
        }),
        ("shape and span disagree", {
            let header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,12]}}";
            let mut f = (header.len() as u64).to_le_bytes().to_vec();
            f.extend_from_slice(header.as_bytes());
            f.extend_from_slice(&[0u8; 12]);
            f
        }),
    ];

    for (what, bytes) in cases {
        let path = temp("fastnn_st_corrupt.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let result = load_safetensors(&Linear::new(2, 2), &path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "{what}: corrupt file should not load");
    }
}
