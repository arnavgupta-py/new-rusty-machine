"""
This module provides the high-level Python API for the rusty-machine library.

It defines scikit-learn-like classes that wrap the low-level, high-performance
Rust and CUDA functions. Users interact with these classes (`LinearRegression`,
`LogisticRegression`) to train models and make predictions.
"""

import numpy as np
import cupy as cp
# Import the core Rust functions compiled via PyO3
import rusty_machine


class LinearRegression:
    """
    GPU-accelerated Linear Regression with L2 (Ridge) regularization.
    
    This model solves for the optimal parameters (theta) using the
    analytical Normal Equation method, which is implemented on the GPU
    using cuBLAS and cuSOLVER for high performance.
    
    The equation solved is: θ = (X^T X + αI)^(-1) X^T y
    
    Parameters:
    -----------
    alpha : float, default=0.0
        L2 regularization strength (lambda). Higher values specify stronger
        regularization. `alpha=0.0` corresponds to standard (unregularized)
        Linear Regression.
    """
    
    def __init__(self, alpha=0.0):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
        This method performs all heavy computation on the GPU.
        1. Adds an intercept (bias) term to the feature matrix X.
        2. Transfers the augmented X and target vector y to the GPU.
        3. Calls the Rust backend `solve_normal_equation_device` to compute
           the model parameters (theta).
        4. Retrieves the parameters and stores them in `self.coef_` and
           `self.intercept_`.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be a NumPy array or a CuPy array.
        y : array-like of shape (n_samples,)
            Target values. Can be a NumPy array or a CuPy array.
            
        Returns:
        --------
        self : object
            The fitted estimator.
        """
        # Ensure data is on the CPU (as NumPy) for preprocessing
        if isinstance(X, np.ndarray):
            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32)
        else:
            # If X is a CuPy array, get it back to host (CPU)
            X_np = X.get()
            y_np = y.get()

        # Add intercept term (column of ones) to X
        ones = np.ones((X_np.shape[0], 1), dtype=np.float32)
        X_b_np = np.hstack([X_np, ones])

        # Transfer data to GPU as contiguous CuPy arrays
        X_b_gpu = cp.asarray(np.ascontiguousarray(X_b_np))
        y_gpu = cp.asarray(y_np)

        samples, features = X_b_gpu.shape
        # Allocate device memory for the output (theta)
        theta_gpu = cp.empty(features, dtype=cp.float32)

        # Call the GPU-accelerated Rust function
        rusty_machine.solve_normal_equation_device(
            X_b_gpu.data.ptr,  # Pass raw device pointer for X
            y_gpu.data.ptr,    # Pass raw device pointer for y
            theta_gpu.data.ptr, # Pass raw device pointer for output theta
            samples,
            features,
            self.alpha
        )

        # Get the computed parameters back from GPU to CPU
        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[-1]
        self.coef_ = theta_host[:-1]

        return self

    def predict(self, X):
        """
        Predict target values using the fitted linear model.
        
        This computation is performed on the CPU using NumPy, as prediction
        is typically a lightweight operation (a single matrix-vector dot product).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")

        # Ensure X is a NumPy array for CPU-based prediction
        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)
        # Calculate prediction: y = X * coef_ + intercept_
        return X_np @ self.coef_ + self.intercept_


class LogisticRegression:
    """
    GPU-accelerated Logistic Regression with L1 or L2 regularization.
    
    This model is trained using Mini-Batch Gradient Descent. All training
    steps (forward pass, error calculation, gradient computation, and
    parameter updates) are executed on the GPU via custom CUDA kernels
    and cuBLAS functions.
    
    Parameters:
    -----------
    epochs : int, default=1000
        Number of passes over the training data.
    lr : float, default=0.01
        Learning rate for gradient descent.
    batch_size : int, default=256
        Number of samples per gradient update.
    penalty : {'l1', 'l2'}, default='l2'
        The type of regularization penalty to apply.
        - 'l1' (Lasso) promotes sparsity in coefficients.
        - 'l2' (Ridge) prevents overfitting by penalizing large coefficients.
    alpha : float, default=0.0
        Regularization strength. Must be a non-negative float.
    random_state : int or None, default=None
        Seed for the random number generator used for data shuffling,
        ensuring reproducible results.
    """

    def __init__(self, epochs=1000, lr=0.01, batch_size=256, penalty='l2', alpha=0.0, random_state=None):
        if penalty not in ['l1', 'l2']:
            raise ValueError("Penalty must be 'l1' or 'l2'")
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
            
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the logistic regression model using GPU-accelerated mini-batch GD.
        
        1. Sets the random seed for reproducibility.
        2. Adds an intercept (bias) term to the feature matrix X.
        3. Shuffles the data on the CPU (a crucial step for mini-batch GD).
        4. Transfers the shuffled data and initial parameters to the GPU.
        5. Calls the Rust backend `train_logistic_minibatch_gpu` to perform
           training for the specified number of epochs.
        6. Retrieves the final parameters and stores them in `self.coef_` and
           `self.intercept_`.
           
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. Should be scaled/normalized for best results.
        y : array-like of shape (n_samples,)
            Target values (0 or 1 for binary classification).
            
        Returns:
        --------
        self : object
            The fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Ensure data is on the CPU (as NumPy) for preprocessing
        if isinstance(X, np.ndarray):
            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32).ravel()
        else:
            # If X is a CuPy array, get it back to host (CPU)
            X_np = X.get()
            y_np = y.get().ravel()

        # Add intercept term (column of ones)
        ones = np.ones((X_np.shape[0], 1), dtype=np.float32)
        X_b_np = np.hstack([X_np, ones])

        samples, features = X_b_np.shape
        # Initialize parameters (theta) to zeros on the CPU
        theta_np = np.zeros(features, dtype=np.float32)

        # Shuffle data for stochastic gradient descent
        permutation = np.random.permutation(samples)
        X_shuffled = np.ascontiguousarray(X_b_np[permutation])
        y_shuffled = np.ascontiguousarray(y_np[permutation])

        # Transfer shuffled data to the GPU
        X_gpu = cp.asarray(X_shuffled)
        y_gpu = cp.asarray(y_shuffled)
        theta_gpu = cp.asarray(theta_np)

        # Map penalty string to integer for the Rust function
        # 1 = 'l1', 2 = 'l2'
        penalty_type = 1 if self.penalty == 'l1' else 2

        # Call the GPU-accelerated Rust function
        rusty_machine.train_logistic_minibatch_gpu(
            X_gpu.data.ptr,
            y_gpu.data.ptr,
            theta_gpu.data.ptr,
            samples,
            features,
            self.epochs,
            self.lr,
            self.batch_size,
            self.alpha,
            penalty_type
        )

        # Get the final trained parameters back from GPU to CPU
        theta_host = theta_gpu.get()
        self.intercept_ = theta_host[-1]
        self.coef_ = theta_host[:-1]
        
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        This computation is performed on the CPU using NumPy.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        proba : ndarray of shape (n_samples, 2)
            The probability of the sample for each class [0, 1].
        """
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() before predict_proba().")

        # Ensure X is a NumPy array for CPU-based prediction
        X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)
        
        # Calculate logits: z = X * coef_ + intercept_
        logits = X_np @ self.coef_ + self.intercept_
        
        # Calculate sigmoid: p(y=1) = 1 / (1 + exp(-z))
        # np.clip is used for numerical stability to avoid overflow in exp()
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.hstack([1 - probs.reshape(-1, 1), probs.reshape(-1, 1)])

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.
        
        This method uses the probabilities from `predict_proba` and
        applies a 0.5 threshold.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels (0 or 1).
        """
        # Get the probability of class 1
        prob_class_1 = self.predict_proba(X)[:, 1]
        # Return 1 if probability > 0.5, else 0
        return (prob_class_1 > 0.5).astype(int)