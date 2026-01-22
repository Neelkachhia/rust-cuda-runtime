# Rust-Managed CUDA Runtime (Vector Addition)

A minimal **Rust + C++ + CUDA** project that demonstrates **safe GPU memory management**, **Rust–CUDA FFI integration**, and **end-to-end GPU computation** using NVIDIA CUDA.

This project is intentionally small but deep, focusing on **systems-level correctness, ownership, and performance** rather than large frameworks.

---

## 🚀 Key Highlights

- **Rust-managed GPU memory** using RAII (`DeviceBuffer<T>`)
- **CUDA kernel execution** via a minimal C++/CUDA layer
- **Safe Rust API with isolated `unsafe` FFI boundary**
- Explicit **host ↔ device memory transfers**
- Fully tested **end-to-end GPU computation**

---

## 🧠 Motivation

CUDA programming in C++ is powerful but error-prone:
- GPU memory leaks
- Use-after-free bugs
- Manual lifetime management

Rust provides:
- Ownership and lifetimes
- Deterministic cleanup (`Drop`)
- Compile-time safety

This project combines **Rust for safety and orchestration** with **CUDA/C++ for raw GPU execution**, following patterns used in real GPU runtimes.

---

## 🏗️ Architecture Overview

Rust (host, safe API)
├── DeviceBuffer<T> // owns GPU memory
├── Host ↔ Device copies
├── Kernel launch (FFI)
│
└── FFI boundary (unsafe)
↓
C++ / CUDA
├── cudaMalloc / cudaFree
├── cudaMemcpy
├── Kernel launcher
└── CUDA kernel
↓
NVIDIA GPU

## 📂 Project Structure

rust_cuda_runtime/
├── src/
│ └── lib.rs # Rust API + DeviceBuffer
├── cuda/
│ ├── vector_add.cu # CUDA kernel + C interface
│ ├── vector_add.h # C header for FFI
│ └── libvector_add.a # CUDA static library
├── tests/
│ └── vector_add.rs # End-to-end GPU test
├── build.rs # Cargo build script (linking)
└── Cargo.toml


## 🔑 Core Components

### 1️⃣ `DeviceBuffer<T>` (Rust)

- Wraps `cudaMalloc` / `cudaFree`
- Enforces **single ownership**
- Automatically frees GPU memory via `Drop`
- Prevents leaks and misuse

```rust
let d_buf = DeviceBuffer::<f32>::new(1024);
// GPU memory freed automatically when dropped

2️⃣ CUDA Kernel (C++)

__global__ void vector_add(const float* a,
                           const float* b,
                           float* c,
                           int n);

Executed on GPU

Launched via a C-compatible wrapper

Synchronized explicitly

3️⃣ Rust ↔ CUDA FFI

extern "C" interface

All unsafety isolated at the boundary

Rust API remains safe and ergonomic

🧪 Testing

An end-to-end test validates:

Host → Device copy

Kernel execution

Device → Host copy

Correct numerical results

Run tests:
cargo test

Expected output:
test test_vector_add ... ok

🛠️ Build & Requirements
Requirements

Linux / WSL2

NVIDIA GPU

CUDA Toolkit (tested with CUDA 12.9)

Rust (stable)

Build

cargo build


🧩 Why This Project Is Small (On Purpose)

This project avoids large frameworks to:

Make every line explainable

Focus on GPU systems fundamentals

Demonstrate engineering judgment

The goal is depth, not breadth.

📌 What This Demonstrates

GPU memory lifecycle management

Host vs device execution model

Rust ownership applied to GPUs

CUDA runtime integration

Build systems & linker knowledge