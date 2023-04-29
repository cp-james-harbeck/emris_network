// ---------------- Helper Functions ----------------

use std::cmp;
use std::collections::HashMap;
use std::f32;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Response, WebGl2RenderingContext as GL};

pub const BUFFER_USAGE_DICT: HashMap<&str, u32> = [
    ("copy_from", GL::COPY_READ_BUFFER),
    ("copy_to", GL::COPY_WRITE_BUFFER),
    ("storage", GL::STORAGE_BUFFER),
    ("uniform", GL::UNIFORM_BUFFER),
    ("map_read", GL::MAP_READ_BUFFER),
]
.iter()
.cloned()
.collect();

pub async fn fetch_bin(url: &str) -> Result<Vec<f32>, JsValue> {
    let window = web_sys::window().unwrap();
    let response = JsFuture::from(window.fetch_with_str(url)).await?;
    let response: Response = response.dyn_into()?;
    let buffer = JsFuture::from(response.array_buffer()?).await?;
    let buffer: js_sys::Float32Array = js_sys::Float32Array::new(&buffer);
    Ok(buffer.to_vec())
}

pub fn wg_size(dim: usize, size: usize) -> usize {
    cmp::min((dim + size - 1) / size, 256)
}

pub fn sample_from_distribution(probs: &[f32]) -> usize {
    let rand = rand::random::<f32>();
    let mut cumulative_prob = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative_prob += prob;
        if rand < cumulative_prob {
            return i;
        }
    }
    probs.len() - 1
}

pub fn cpu_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits
        .iter()
        .map(|&logit| ((logit - max_logit) / temperature).exp())
        .collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    exp_logits
        .into_iter()
        .map(|exp_logit| exp_logit / sum_exp_logits)
        .collect()
}

pub fn select_top_k(probs: &[f32], top_k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut sorted_indices: Vec<(usize, &f32)> = probs.iter().enumerate().collect();
    sorted_indices.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top_k_indices: Vec<usize> = sorted_indices
        .iter()
        .map(|&(index, _)| index)
        .take(top_k)
        .collect();
    let top_k_probs: Vec<f32> = top_k_indices.iter().map(|&index| probs[index]).collect();
    (top_k_indices, top_k_probs)
}

// ----------------------- Matrix Operations -----------------------

pub fn transpose(
    array: &[f32],
    input_rows: usize,
    input_cols: usize,
) -> Result<Vec<f32>, &'static str> {
    if array.len() != input_rows * input_cols {
        return Err("Transpose dims failed");
    }

    let mut transpose = vec![];
    for col in 0..input_cols {
        for row in 0..input_rows {
            transpose.push(array[row * input_cols + col]);
        }
    }

    Ok(transpose)
}

pub fn least_prime_factor(n: usize, start: usize) -> usize {
    for i in start..=((n as f64).sqrt() as usize) {
        if n % i == 0 {
            return i;
        }
    }
    n
}

pub fn format_as_matrix(float_array: &[f32], dim_a: usize, dim_b: usize) -> Vec<Vec<f32>> {
    let mut result_matrix = vec![];
    for i in 0..dim_a {
        result_matrix.push(float_array[i * dim_b..(i + 1) * dim_b].to_vec());
    }
    result_matrix
}
