// This file contains the custom CUDA kernels used by the rusty-machine library.
// These kernels are compiled into PTX (Parallel Thread eXecution) assembly
// and loaded by the Rust backend to perform high-speed, parallel computations
// on the GPU.

/**
 * @brief Transposes a matrix (input) into (output).
 * * This is a standard, tile-less transpose kernel. Each thread is
 * responsible for copying one element from the input matrix at (y, x)
 * to the output matrix at (x, y).
 * * @param input  Pointer to the input matrix (rows x cols) on the device.
 * @param output Pointer to the output matrix (cols x rows) on the device.
 * @param rows   The number of rows in the input matrix.
 * @param cols   The number of columns in the input matrix.
 */
extern "C" __global__ void transpose(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // Calculate the global 2D thread indices (x, y)
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    
    // Boundary check: ensure the thread is within the matrix dimensions
    if (x < cols && y < rows) {
        // Perform the transpose: output[x][y] = input[y][x]
        output[x * rows + y] = input[y * cols + x];
    }
}

/**
 * @brief Fused kernel for Logistic Regression error calculation.
 * * This kernel performs two operations in one pass to save memory bandwidth:
 * 1. Computes the sigmoid function: sigmoid(z) = 1 / (1 + exp(-z))
 * 2. Subtracts the true label y: error = sigmoid(z) - y
 * * Clipping (fmaxf/fminf) is used to prevent `expf` from overflowing or
 * underflowing, ensuring numerical stability.
 *
 * @param logits Pointer to the input vector of logits (z = X * theta).
 * @param y_true Pointer to the vector of true labels (y).
 * @param output Pointer to the output vector to store the (sigmoid - y) error.
 * @param n      The number of elements in the vectors (batch size).
 */
extern "C" __global__ void fused_sigmoid_sub(
    const float* logits,
    const float* y_true,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float z = logits[idx];
        // Clip to prevent overflow/underflow in expf()
        z = fmaxf(-20.0f, fminf(20.0f, z));
        float sigmoid = 1.0f / (1.0f + expf(-z));
        output[idx] = sigmoid - y_true[idx];
    }
}

/**
 * @brief Adds the L2 regularization term (alpha) to the diagonal of a matrix.
 * * This is used in Ridge Regression to compute (X^T*X + αI).
 * It adds `alpha` to `matrix[i][i]` for all `i`.
 * * @param matrix Pointer to the matrix (e.g., X^T*X) on the device.
 * @param alpha  The L2 regularization strength.
 * @param n      The dimension of the square matrix (n x n).
 */
extern "C" __global__ void add_regularization_term(
    float* matrix,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // We stop at n-1 because the last feature is the bias/intercept term,
    // which is typically not regularized.
    if (idx < n - 1) {  
        // Add alpha to the diagonal element: matrix[idx * n + idx]
        matrix[idx * n + idx] += alpha;
    }
}

/**
 * @brief Performs the gradient descent update step for L2 (Ridge) regularization.
 * * The update rule is:
 * θ_new = θ_old - lr * ( (grad / batch_size) + (α * θ_old / batch_size) )
 * * The gradient `grad` and the regularization term `α * θ` are both
 * scaled by the `batch_size` to ensure the learning rate `lr` is
 * independent of the batch size.
 *
 * @param theta      Pointer to the model parameters (theta) vector.
 * @param grad       Pointer to the gradient vector (X^T * error).
 * @param lr         The learning rate.
 * @param alpha_reg  The L2 regularization strength (α).
 * @param features   The number of features (dimension of theta).
 * @param batch_size The size of the mini-batch.
 */
extern "C" __global__ void l2_regularized_update(
    float* theta, 
    const float* grad, 
    float lr, 
    float alpha_reg, 
    int features, 
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        // Gradient component scaled by batch size
        float grad_component = grad[idx] / batch_size;
        
        float reg_component = 0.0f;
        // Don't regularize the bias term (the last element)
        if (idx < features - 1) {
            // L2 regularization component, also scaled by batch size
            reg_component = (alpha_reg / batch_size) * theta[idx];
        }
        
        // Perform the update
        theta[idx] -= lr * (grad_component + reg_component);
    }
}

/**
 * @brief Performs the gradient descent update step for L1 (Lasso) regularization.
 * * This implements the Proximal Gradient Descent update, specifically using
 * the Soft Thresholding operator.
 * * 1. θ_temp = θ_old - lr * (grad / batch_size)
 * 2. threshold = lr * α / (batch_size * batch_size)
 * 3. θ_new = soft_threshold(θ_temp, threshold)
 * * soft_threshold(a, t) = sign(a) * max(0, |a| - t)
 * This is implemented as:
 * if (a > t):  a - t
 * if (a < -t): a + t
 * else:       0.0
 *
 * @param theta      Pointer to the model parameters (theta) vector.
 * @param grad       Pointer to the gradient vector (X^T * error).
 * @param lr         The learning rate.
 * @param alpha_reg  The L1 regularization strength (α).
 * @param features   The number of features (dimension of theta).
 * @param batch_size The size of the mini-batch.
 */
extern "C" __global__ void l1_regularized_update(
    float* theta, 
    const float* grad, 
    float lr, 
    float alpha_reg, 
    int features, 
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        // 1. Standard gradient step
        float grad_component = grad[idx] / batch_size;
        float theta_new = theta[idx] - lr * grad_component;
        
        // 2. Apply soft thresholding (L1 proximal operator)
        // Don't regularize the bias term (the last element)
        if (idx < features - 1) {
            // 2a. Calculate the threshold
            // This scaling is non-standard but matches the original logic.
            // A more common threshold is `lr * alpha_reg / batch_size`.
            float threshold = lr * alpha_reg / (batch_size * batch_size);
            
            // 2b. Apply the soft thresholding operation
            if (theta_new > threshold) {
                theta[idx] = theta_new - threshold;
            } else if (theta_new < -threshold) {
                theta[idx] = theta_new + threshold;
            } else {
                // Set weights to zero if they are within the threshold
                theta[idx] = 0.0f;
            }
        } else {
            // No regularization for the bias term
            theta[idx] = theta_new;
        }
    }
}