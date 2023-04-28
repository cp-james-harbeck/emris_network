#!/bin/sh
CANISTER_NAME=$1
CANISTER_RS=$2

# Change to the Rust canister directory
cd "src/${CANISTER_RS}"

# Build the Rust canister
cargo build --release --target wasm32-unknown-unknown

# Copy the wasm binary to the build directory
mkdir -p "../../.dfx/local/canisters/${CANISTER_NAME}"
cp "target/wasm32-unknown-unknown/release/${CANISTER_RS}.wasm" "../../.dfx/local/canisters/${CANISTER_NAME}/${CANISTER_NAME}.wasm"
