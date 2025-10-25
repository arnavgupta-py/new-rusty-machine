// This is the build script for the `rusty_machine` Rust crate.
// Its primary job is to compile the custom CUDA kernels (.cu) into
// PTX (Parallel Thread eXecution) assembly, which will be embedded
// into the final Rust library. It also links the necessary NVIDIA libraries.

use std::process::Command;

fn main() {
    // --- 1. Link NVIDIA CUDA Libraries ---
    // Instruct cargo to link against the core CUDA runtime library.
    println!("cargo:rustc-link-lib=cuda");
    // Link against the cuSOLVER library for dense linear algebra (e.g., matrix inversion).
    println!("cargo:rustc-link-lib=cusolver");
    // Link against the cuBLAS library for Basic Linear Algebra Subprograms (e.g., matrix multiplication).
    println!("cargo:rustc-link-lib=cublas");

    // Provide the linker with the search path for the CUDA libraries.
    // This is the standard location on most Linux systems.
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // --- 2. Compile CUDA Kernels ---
    // Execute the NVIDIA CUDA Compiler (nvcc) to build our custom kernels.
    let status = Command::new("nvcc")
        // Apply optimizations.
        .arg("-O3")
        // Target the 'sm_86' architecture (NVIDIA Ampere). This is a good
        // balance of performance and compatibility with modern GPUs (like the
        // RTX 4060 used in development, which supports sm_89 but is
        // backward-compatible with sm_86).
        .arg("-arch=sm_86")
        // Specify the output format as PTX assembly.
        .arg("-ptx")
        // Set the output file path.
        .arg("-o")
        .arg("src/kernels.ptx")
        // Specify the input CUDA source file.
        .arg("src/kernels.cu")
        .status()
        .expect("Failed to execute nvcc. Ensure the CUDA toolkit is installed and nvcc is in your PATH.");

    // If the `nvcc` command fails, panic the build process with an error.
    if !status.success() {
        panic!("nvcc failed to compile the CUDA kernel.");
    }

    // --- 3. Register Dependencies ---
    // Tell cargo to re-run this build script only if the source CUDA
    // kernel file has changed. This prevents unnecessary recompilation.
    println!("cargo:rerun-if-changed=src/kernels.cu");
}