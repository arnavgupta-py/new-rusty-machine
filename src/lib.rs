//! This module provides the core Rust implementation and Python bindings for `rusty_machine`.
//! It leverages CUDA, cuBLAS, and cuSOLVER to provide GPU-accelerated
//! machine learning algorithms to a Python frontend.

use cust::prelude::*;
use cust::memory::{DeviceBuffer, DevicePointer, DeviceMemory};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::OnceLock;
use std::os::raw::{c_int, c_void};

/// This module defines the raw Foreign Function Interface (FFI) bindings
/// to the NVIDIA cuBLAS and cuSOLVER libraries. This allows Rust to call
/// the highly optimized C functions provided by the CUDA toolkit.
mod ffi {
    use super::{c_int, c_void};

    // --- cuBLAS Definitions ---

    #[repr(C)]
    pub struct CublasContext { _private: [u8; 0] }
    pub type CublasHandle = *mut CublasContext;

    #[repr(C)]
    pub struct CusolverDnContext { _private: [u8; 0] }
    pub type CusolverDnHandle = *mut CusolverDnContext;

    /// Constant representing a non-transpose operation in cuBLAS.
    pub const CUBLAS_OP_N: c_int = 0;
    /// Constant representing a transpose operation in cuBLAS.
    pub const CUBLAS_OP_T: c_int = 1;

    #[link(name = "cublas")]
    extern "C" {
        pub fn cublasCreate_v2(handle: *mut CublasHandle) -> c_int;
        pub fn cublasDestroy_v2(handle: CublasHandle) -> c_int;
        pub fn cublasSetStream_v2(handle: CublasHandle, stream: *mut c_void) -> c_int;
        /// Configures cuBLAS to use Tensor Cores if available.
        pub fn cublasSetMathMode(handle: CublasHandle, mode: c_int) -> c_int;
        
        /// cuBLAS FFI binding for Single-precision General Matrix-Vector multiplication (SGEMV).
        /// y = alpha * A * x + beta * y
        pub fn cublasSgemv_v2(
            handle: CublasHandle,
            trans: c_int,
            m: c_int, n: c_int,
            alpha: *const f32,
            A: *const c_void, lda: c_int,
            x: *const c_void, incx: c_int,
            beta: *const f32,
            y: *mut c_void, incy: c_int,
        ) -> c_int;
        
        /// cuBLAS FFI binding for Single-precision General Matrix-Matrix multiplication (SGEMM).
        /// C = alpha * A * B + beta * C
        pub fn cublasSgemm_v2(
            handle: CublasHandle,
            transa: c_int, transb: c_int,
            m: c_int, n: c_int, k: c_int,
            alpha: *const f32,
            A: *const c_void, lda: c_int,
            B: *const c_void, ldb: c_int,
            beta: *const f32,
            C: *mut c_void, ldc: c_int
        ) -> c_int;
    }

    // --- cuSOLVER Definitions ---

    #[link(name = "cusolver")]
    extern "C" {
        pub fn cusolverDnCreate(handle: *mut CusolverDnHandle) -> c_int;
        pub fn cusolverDnDestroy(handle: CusolverDnHandle) -> c_int;
        pub fn cusolverDnSetStream(handle: CusolverDnHandle, stream: *mut c_void) -> c_int;
        
        /// cuSOLVER FFI binding to get the required workspace size for SGETRF (LU decomposition).
        pub fn cusolverDnSgetrf_bufferSize(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int) -> c_int;
        /// cuSOLVER FFI binding for SGETRF (Single-precision LU decomposition).
        pub fn cusolverDnSgetrf(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int) -> c_int;
        /// cuSOLVER FFI binding for SGETRS (Single-precision LU solver).
        pub fn cusolverDnSgetrs(handle: CusolverDnHandle, trans: c_int, n: c_int, nrhs: c_int, A: *const f32, lda: c_int, devIpiv: *const c_int, B: *mut f32, ldb: c_int, devInfo: *mut c_int) -> c_int;
    }
}

/// A wrapper struct for the cuBLAS handle to make it thread-safe (Send + Sync).
/// This is necessary for storing it in the `OnceLock` global static.
struct SyncCublasHandle(ffi::CublasHandle);
unsafe impl Send for SyncCublasHandle {}
unsafe impl Sync for SyncCublasHandle {}

/// A wrapper struct for the cuSOLVER handle to make it thread-safe (Send + Sync).
/// This is necessary for storing it in the `OnceLock` global static.
struct SyncCusolverHandle(ffi::CusolverDnHandle);
unsafe impl Send for SyncCusolverHandle {}
unsafe impl Sync for SyncCusolverHandle {}

/// Encapsulates all necessary GPU resources.
/// This includes the CUDA context, execution stream, loaded custom kernels (module),
/// and handles to the cuBLAS and cuSOLVER libraries.
struct GpuContext {
    _ctx: Context,
    stream: Stream,
    module: Module,
    cublas_handle: SyncCublasHandle,
    cusolver_handle: SyncCusolverHandle,
}

/// Implements the `Drop` trait for `GpuContext` to ensure that all
/// GPU resources (library handles) are properly released when the context
/// goes out of scope, preventing memory leaks.
impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasDestroy_v2(self.cublas_handle.0);
            ffi::cusolverDnDestroy(self.cusolver_handle.0);
        }
    }
}

/// A thread-safe, lazily-initialized static singleton for the `GpuContext`.
/// This ensures that the GPU is initialized only once, and the same context
/// is shared across all Python function calls.
static GLOBAL_CTX: OnceLock<GpuContext> = OnceLock::new();

/// A utility function to convert any error type that implements `std::fmt::Display`
/// into a Python `PyRuntimeError`.
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Retrieves (or initializes) the global `GpuContext` singleton.
/// On first call, this function:
/// 1. Initializes the CUDA driver.
/// 2. Selects the primary GPU device (device 0).
/// 3. Creates a new CUDA context and a non-blocking stream.
/// 4. Loads the compiled PTX (kernels.ptx) file into a `Module`.
/// 5. Creates cuBLAS and cuSOLVER handles.
/// 6. Associates the handles with the CUDA stream.
/// 7. Enables Tensor Core operations in cuBLAS.
/// 8. Stores the initialized context in the `GLOBAL_CTX` `OnceLock`.
/// Subsequent calls will efficiently return the existing context.
fn get_gpu_context() -> PyResult<&'static GpuContext> {
    if let Some(ctx) = GLOBAL_CTX.get() {
        return Ok(ctx);
    }
    let created = (|| -> Result<GpuContext, PyErr> {
        cust::init(CudaFlags::empty()).map_err(to_py_err)?;
        let device = Device::get_device(0).map_err(to_py_err)?;
        let ctx = Context::new(device).map_err(to_py_err)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(to_py_err)?;
        let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;

        let mut cublas_handle: ffi::CublasHandle = std::ptr::null_mut();
        unsafe { 
            ffi::cublasCreate_v2(&mut cublas_handle);
            ffi::cublasSetStream_v2(cublas_handle, stream.as_inner() as *mut c_void);
            ffi::cublasSetMathMode(cublas_handle, 1); // Enable Tensor Cores
        }

        let mut cusolver_handle: ffi::CusolverDnHandle = std::ptr::null_mut();
        unsafe { 
            ffi::cusolverDnCreate(&mut cusolver_handle);
            ffi::cusolverDnSetStream(cusolver_handle, stream.as_inner() as *mut c_void);
        }

        Ok(GpuContext { _ctx: ctx, stream, module, cublas_handle: SyncCublasHandle(cublas_handle), cusolver_handle: SyncCusolverHandle(cusolver_handle) })
    })();
    match created {
        Ok(ctx_val) => {
            GLOBAL_CTX.set(ctx_val).map_err(|_| to_py_err("Failed to set GLOBAL_CTX"))?;
            Ok(GLOBAL_CTX.get().expect("GLOBAL_CTX set but not present"))
        }
        Err(e) => Err(e),
    }
}

/// A helper function to safely reconstruct a `DevicePointer<f32>`
/// from a raw `u64` pointer address passed from Python (e.g., from a Cupy array).
fn device_ptr_from_u64(ptr: u64) -> DevicePointer<f32> {
    DevicePointer::from_raw(ptr)
}

/// A Python-exposed function that performs a GPU-accelerated matrix transpose.
/// It launches the custom "transpose" CUDA kernel.
///
/// # Arguments
/// * `in_ptr` - The raw `u64` device pointer to the input matrix.
/// * `out_ptr` - The raw `u64` device pointer to the output matrix.
/// * `m` - The number of rows in the input matrix.
/// * `n` - The number of columns in the input matrix.
#[pyfunction]
fn gpu_transpose(in_ptr: u64, out_ptr: u64, m: usize, n: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let func = ctx.module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    let stream = &ctx.stream;
    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            device_ptr_from_u64(in_ptr), device_ptr_from_u64(out_ptr), m as i32, n as i32
        )).map_err(to_py_err)?;
    }
    Ok(())
}

/// A Python-exposed function that performs a GPU-accelerated matrix inversion
/// (in-place on `a_ptr`).
/// This function uses the cuSOLVER library to perform an LU decomposition
/// (`cusolverDnSgetrf`) followed by solving against an identity matrix
/// (`cusolverDnSgetrs`) to find the inverse.
///
/// # Arguments
/// * `a_ptr` - The raw `u64` device pointer to the input matrix (will be overwritten).
/// * `inv_ptr` - The raw `u64` device pointer to store the resulting inverse.
/// * `n` - The dimension of the square matrix (n x n).
#[pyfunction]
fn gpu_inverse(a_ptr: u64, inv_ptr: u64, n: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let mut identity = vec![0f32; n * n];
    for i in 0..n { identity[i * n + i] = 1.0f32; }
    let inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;

    unsafe {
        let handle = ctx.cusolver_handle.0;
        let ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        let a_dev_ptr = a_ptr as *mut f32;
        
        // 1. Get workspace size for LU decomposition
        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork);
        let work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;
        
        // 2. Perform LU decomposition in-place on `a_dev_ptr`
        ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut i32, info_dev.as_raw_ptr() as *mut i32);
        
        // 3. Solve A * X = I (where I is the identity) to find X = A^-1
        //    The result is stored in `inv_dev_temp`.
        ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev_ptr as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const i32, inv_dev_temp.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut i32);

        // 4. Copy the result from the temporary buffer to the final output pointer
        let mut inv_out_slice = cust::memory::DeviceSlice::from_raw_parts_mut(device_ptr_from_u64(inv_ptr), n * n);
        inv_dev_temp.copy_to(&mut inv_out_slice).map_err(to_py_err)?;
    }
    Ok(())
}

/// A Python-exposed function to solve for the parameters `theta` of Linear
/// Regression using the Normal Equation: θ = (X^T*X + αI)^-1 * (X^T*y).
/// All computations are performed on the GPU.
///
/// # Arguments
/// * `x_ptr` - Device pointer to the feature matrix `X` (samples x features).
/// * `y_ptr` - Device pointer to the target vector `y` (samples x 1).
/// * `theta_ptr` - Device pointer to store the resulting `theta` vector (features x 1).
/// * `samples` - The number of training samples.
/// * `features` - The number of features.
/// * `alpha_reg` - The L2 regularization strength (Ridge regression).
#[pyfunction]
fn solve_normal_equation_device(x_ptr: u64, y_ptr: u64, theta_ptr: u64, samples: usize, features: usize, alpha_reg: f32) -> PyResult<()> {
    let ctx = get_gpu_context()?;

    let alpha = 1.0f32;
    let beta = 0.0f32;
    let handle = ctx.cublas_handle.0;
    let features_i32 = features as i32;
    let samples_i32 = samples as i32;

    let x_dev_ptr = x_ptr as *const c_void;
    let y_dev_ptr = y_ptr as *const c_void;
    let theta_dev_ptr = theta_ptr as *mut c_void;

    let stream = &ctx.stream;

    unsafe {
        // Allocate temporary device buffers
        let xtx_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let mut xtx_inv_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let xt_y_dev = DeviceBuffer::<f32>::uninitialized(features).map_err(to_py_err)?;

        let xtx_dev_ptr = xtx_dev.as_raw_ptr() as *mut c_void;
        let xtx_inv_dev_ptr = xtx_inv_dev.as_raw_ptr() as *mut c_void;
        let xt_y_dev_ptr = xt_y_dev.as_raw_ptr() as *mut c_void;

        // 1. Compute X^T * X (using cuBLAS SGEMM)
        ffi::cublasSgemm_v2(
            handle, 
            ffi::CUBLAS_OP_N, ffi::CUBLAS_OP_T,
            features_i32, features_i32, samples_i32,
            &alpha,
            x_dev_ptr, features_i32,
            x_dev_ptr, features_i32,
            &beta,
            xtx_dev_ptr, features_i32
        );

        // 2. Add regularization term αI to X^T*X
        if alpha_reg > 0.0 {
            let add_reg_func = ctx.module.get_function("add_regularization_term").map_err(to_py_err)?;
            let block_dim = 256u32;
            let grid_dim = (features as u32 + block_dim - 1) / block_dim;
            launch!(add_reg_func<<<grid_dim, block_dim, 0, stream>>>(
                xtx_dev.as_device_ptr(), alpha_reg, features_i32
            )).map_err(to_py_err)?;
        }

        // 3. Copy (X^T*X + αI) for inversion
        xtx_dev.copy_to(&mut xtx_inv_dev).map_err(to_py_err)?;
        
        // 4. Invert (X^T*X + αI)
        gpu_inverse(xtx_inv_dev.as_raw_ptr() as u64, xtx_inv_dev.as_raw_ptr() as u64, features)?;

        // 5. Compute X^T * y (using cuBLAS SGEMV)
        ffi::cublasSgemv_v2(
            handle, 
            ffi::CUBLAS_OP_N,
            features_i32, samples_i32,
            &alpha,
            x_dev_ptr, features_i32,
            y_dev_ptr, 1,
            &beta,
            xt_y_dev_ptr, 1
        );

        // 6. Compute Theta = (X^T*X + αI)^-1 * (X^T*y)
        ffi::cublasSgemv_v2(
            handle, 
            ffi::CUBLAS_OP_N,
            features_i32, features_i32,
            &alpha,
            xtx_inv_dev_ptr, features_i32,
            xt_y_dev_ptr, 1,
            &beta,
            theta_dev_ptr, 1
        );
    }
    
    // Wait for all asynchronous operations in the stream to complete
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// A Python-exposed function to train a Logistic Regression model using
/// GPU-accelerated Mini-Batch Gradient Descent.
///
/// # Arguments
/// * `x_ptr` - Device pointer to the feature matrix `X`.
/// * `y_ptr` - Device pointer to the target vector `y`.
/// * `theta_ptr` - Device pointer to the `theta` vector (parameters).
/// * `samples` - Total number of training samples.
/// * `features` - Number of features (including bias).
/// * `epochs` - Number of passes over the entire dataset.
/// * `lr` - The learning rate.
/// * `batch_size` - The number of samples in each mini-batch.
/// * `alpha_reg` - The regularization strength.
/// * `penalty_type` - An integer specifying regularization type (1 for L1, 2 for L2).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_logistic_minibatch_gpu(
    x_ptr: u64, y_ptr: u64, theta_ptr: u64,
    samples: usize, features: usize, epochs: usize,
    lr: f32, batch_size: usize, alpha_reg: f32,
    penalty_type: i32
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let stream = &ctx.stream;

    // Load custom CUDA kernels
    let fused_sigmoid_sub_f = ctx.module.get_function("fused_sigmoid_sub").map_err(to_py_err)?;
    
    let update_kernel_name = if penalty_type == 1 { 
        "l1_regularized_update" 
    } else { 
        "l2_regularized_update" 
    };
    let reg_update_f = ctx.module.get_function(update_kernel_name).map_err(to_py_err)?;

    // Reconstruct device pointers from u64
    let x_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(x_ptr);
    let y_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(y_ptr);
    let theta_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(theta_ptr);

    // Allocate temporary device buffers once
    let max_batch_size = batch_size;
    let (z_dev, error_dev, grad_dev) = unsafe {
        (
            DeviceBuffer::<f32>::uninitialized(max_batch_size).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(max_batch_size).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(features).map_err(to_py_err)?,
        )
    };

    let alpha = 1.0f32;
    let beta = 0.0f32;
    let features_i32 = features as i32;

    let num_batches = (samples + batch_size - 1) / batch_size;

    // Pre-calculate kernel launch configurations
    let sigmoid_block_dim = 256u32;
    let update_block_dim = 256u32;
    let update_grid_dim = (features as u32 + update_block_dim - 1) / update_block_dim;

    for _e in 0..epochs {
        for i in 0..num_batches {
            let current_batch_size = if i == num_batches - 1 {
                samples - i * batch_size
            } else {
                batch_size
            };
            let current_batch_size_i32 = current_batch_size as i32;

            // Calculate pointers for the current batch
            let x_batch_ptr = unsafe { x_dev_ptr.offset((i * batch_size * features) as isize) };
            let y_batch_ptr = unsafe { y_dev_ptr.offset((i * batch_size) as isize) };

            unsafe {
                // 1. Z = X_batch * theta (cuBLAS SGEMV)
                ffi::cublasSgemv_v2(
                    ctx.cublas_handle.0,
                    ffi::CUBLAS_OP_T,
                    features_i32,
                    current_batch_size_i32,
                    &alpha,
                    x_batch_ptr.as_raw() as *const c_void,
                    features_i32,
                    theta_dev_ptr.as_raw() as *const c_void,
                    1,
                    &beta,
                    z_dev.as_device_ptr().as_raw() as *mut c_void,
                    1,
                );

                // 2. Error = Sigmoid(Z) - Y_batch (Custom CUDA Kernel)
                let sigmoid_grid_dim = (current_batch_size as u32 + sigmoid_block_dim - 1) / sigmoid_block_dim;
                launch!(fused_sigmoid_sub_f<<<sigmoid_grid_dim, sigmoid_block_dim, 0, stream>>>(
                    z_dev.as_device_ptr(), y_batch_ptr, error_dev.as_device_ptr(), current_batch_size_i32
                )).map_err(to_py_err)?;

                // 3. Gradient = X_batch^T * Error (cuBLAS SGEMV)
                ffi::cublasSgemv_v2(
                    ctx.cublas_handle.0,
                    ffi::CUBLAS_OP_N,
                    features_i32,
                    current_batch_size_i32,
                    &alpha,
                    x_batch_ptr.as_raw() as *const c_void,
                    features_i32,
                    error_dev.as_device_ptr().as_raw() as *const c_void,
                    1,
                    &beta,
                    grad_dev.as_device_ptr().as_raw() as *mut c_void,
                    1,
                );

                // 4. Update theta with gradient and regularization (Custom CUDA Kernel)
                launch!(reg_update_f<<<update_grid_dim, update_block_dim, 0, stream>>>(
                    theta_dev_ptr,
                    grad_dev.as_device_ptr(),
                    lr,
                    alpha_reg,
                    features_i32,
                    current_batch_size_i32
                )).map_err(to_py_err)?;
            }
        }
    }
    
    // Wait for all epochs to complete
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

/// Defines the Python module `rusty_machine`.
/// This function is called by the Python interpreter when the module is imported.
/// It registers all the `#[pyfunction]` functions, making them available
/// to Python code (e.g., `rusty_machine.solve_normal_equation_device(...)`).
#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_normal_equation_device, m)?)?;
    m.add_function(wrap_pyfunction!(train_logistic_minibatch_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_inverse, m)?)?;
    Ok(())
}