// webgpu_tokenizer.js

import * as wasm from './pkg'; // Import the WASM module
import init, { tokenize, decode, init_tokenizer } from './pkg/tokenizer.js';

// Initialize the WebGPUCompute object
const webGPUCompute = await wasm.WebGPUCompute.new();

// Initialize the WebAssembly module and tokenizer
await init();
const tokenizer = init_tokenizer("path/to/bpe_model.json", "path/to/bpe_vocab.json");

// Define a function to run the WebGPU computation
async function runGpuComputation(inputData) {
  return await webGPUCompute.run_gpu_computation(inputData);
}

// Define a function to tokenize text
function tokenizeText(text) {
  return tokenize(tokenizer, text);
}

// Define a function to decode token IDs
function decodeTokens(tokenIds) {
  return decode(tokenizer, tokenIds);
}

export { runGpuComputation, tokenizeText, decodeTokens };
