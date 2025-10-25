"""
Benchmark Script: Rusty Machine vs. Scikit-learn

This script provides a comprehensive performance and accuracy comparison
between the GPU-accelerated 'rusty-machine' library and the CPU-bound
'scikit-learn' library.

It executes three distinct benchmark tests:
1.  **Ridge Regression (L2):** Compares `RustyLinearRegression` against
    `sklearn.linear_model.Ridge`.
2.  **Logistic Regression (L2):** Compares `RustyLogisticRegression(penalty='l2')`
    against `sklearn.linear_model.LogisticRegression(penalty='l2')`.
3.  **Logistic Regression (L1):** Compares `RustyLogisticRegression(penalty='l1')`
    against `sklearn.linear_model.LogisticRegression(penalty='l1')`.

For each test, it generates a large synthetic dataset, trains both models
on the same data, and reports the training time, model score (R² or Accuracy),
and the performance speedup.
"""

import time
import numpy as np
from sklearn.datasets import make_regression, make_classification
# Import scikit-learn models for comparison
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our GPU-accelerated models from the 'rustymachine_api'
try:
    from rustymachine_api.models import LinearRegression as RustyLinearRegression
    from rustymachine_api.models import LogisticRegression as RustyLogisticRegression
except ImportError:
    print("Error: 'rusty-machine' library not found.")
    print("Please ensure you have run 'maturin develop --release' in your environment.")
    exit(1)


# --- Benchmark Constants ---

# Regularization strength to be used for all tests
ALPHA = 0.1
# Random seed for reproducible data generation and model training
RANDOM_STATE = 42

# --- Formatting Helper Functions ---

def print_header(title):
    """
    Prints a standardized, centered header to the console.
    """
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def print_results(model_name, score, duration, metric_name, speedup=None):
    """
    Prints a formatted row for the results table.
    
    Parameters:
    -----------
    model_name : str
        The name of the model being reported.
    score : float
        The accuracy score (e.g., R² or Accuracy).
    duration : float
        The time taken for training, in seconds.
    metric_name : str
        The name of the metric (e.g., "R² Score").
    speedup : float, optional
        The calculated performance gain (e.g., 20.0x).
    """
    score_str = f"{score:.4f}"
    duration_str = f"{duration:.4f}s"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "1.00x"
    print(f"{model_name:<30} | {metric_name:<15} | {score_str:<10} | {duration_str:<12} | {speedup_str:<10}")


# --- Benchmark 1: Ridge Regression ---

def run_linear_regression_benchmark():
    """
    Runs the benchmark for L2-regularized Linear Regression (Ridge).
    
    This function compares the performance of `RustyLinearRegression`
    (which uses the GPU-accelerated Normal Equation) against
    `sklearn.linear_model.Ridge`.
    """
    print_header(f"BENCHMARK 1: RIDGE REGRESSION (L2, alpha={ALPHA})")

    # 1. Define dataset parameters
    SAMPLES = 1_000_000
    FEATURES = 100
    INFORMATIVE_FEATURES = 75

    print(f"Generating data: {SAMPLES:,} samples, {FEATURES} features...\n")
    
    # 2. Generate and split data
    X, y = make_regression(
        n_samples=SAMPLES,
        n_features=FEATURES,
        n_informative=INFORMATIVE_FEATURES,
        noise=25,
        random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 3. Train Rusty Machine (GPU)
    print("Training Rusty Machine Ridge (GPU)...")
    model_rm = RustyLinearRegression(alpha=ALPHA)
    start_rm = time.time()
    model_rm.fit(X_train, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test)
    score_rm = r2_score(y_test, preds_rm)
    print(f"Training finished in {duration_rm:.4f} seconds.")

    # 4. Train Scikit-learn (CPU)
    print("\nTraining Scikit-learn Ridge (CPU)...")
    # 'auto' solver is typically fastest for dense data
    model_sk = SklearnRidge(alpha=ALPHA, solver='auto', random_state=RANDOM_STATE)
    start_sk = time.time()
    model_sk.fit(X_train, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test)
    score_sk = r2_score(y_test, preds_sk)
    print(f"Training finished in {duration_sk:.4f} seconds.")

    # 5. Print comparative results
    print("\n" + "-" * 80)
    print(f"{'Model':<30} | {'Metric':<15} | {'R² Score':<10} | {'Train Time':<12} | {'Speedup':<10}")
    print("-" * 80)
    print_results("Rusty Machine Ridge (GPU)", score_rm, duration_rm, "R² Score", speedup=duration_sk / duration_rm if duration_rm > 0 else 0)
    print_results("Scikit-learn Ridge (CPU)", score_sk, duration_sk, "R² Score")
    print("-" * 80 + "\n")


# --- Benchmark 2: Logistic Regression (L2) ---

def run_logistic_l2_regression_benchmark():
    """
    Runs the benchmark for L2-regularized Logistic Regression.
    
    This function compares `RustyLogisticRegression` (using GPU-accelerated
    Mini-Batch Gradient Descent) against `sklearn.linear_model.LogisticRegression`
    (using the 'saga' CPU solver).
    """
    print_header(f"BENCHMARK 2: LOGISTIC REGRESSION (L2 penalty, alpha={ALPHA})")

    # 1. Define dataset and model parameters
    SAMPLES = 500_000
    FEATURES = 100
    INFORMATIVE_FEATURES = 50
    EPOCHS = 100
    LR = 0.05
    BATCH_SIZE = 4096

    print(f"Generating data: {SAMPLES:,} samples, {FEATURES} features...\n")

    # 2. Generate, split, and scale data
    # Scaling is crucial for gradient-based optimization
    X, y = make_classification(
        n_samples=SAMPLES,
        n_features=FEATURES,
        n_informative=INFORMATIVE_FEATURES,
        n_redundant=10,
        random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Rusty Machine (GPU)
    print("Training Rusty Machine Logistic L2 (GPU)...")
    model_rm = RustyLogisticRegression(
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, penalty='l2', alpha=ALPHA, random_state=RANDOM_STATE
    )
    start_rm = time.time()
    model_rm.fit(X_train_scaled, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test_scaled)
    score_rm = accuracy_score(y_test, preds_rm)
    print(f"Training finished in {duration_rm:.4f} seconds.")

    # 4. Train Scikit-learn (CPU)
    print("\nTraining Scikit-learn Logistic L2 (CPU)...")
    # Convert alpha to sklearn's C (C = 1/alpha)
    c_param = 1.0 / ALPHA if ALPHA > 0 else float('inf')
    # Use 'saga' solver as it's fast and supports L2
    model_sk = SklearnLogisticRegression(
        penalty='l2', C=c_param, solver='saga', max_iter=EPOCHS, tol=1e-3, random_state=RANDOM_STATE
    )
    start_sk = time.time()
    model_sk.fit(X_train_scaled, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test_scaled)
    score_sk = accuracy_score(y_test, preds_sk)
    print(f"Training finished in {duration_sk:.4f} seconds.")

    # 5. Print comparative results
    print("\n" + "-" * 80)
    print(f"{'Model':<30} | {'Metric':<15} | {'Accuracy':<10} | {'Train Time':<12} | {'Speedup':<10}")
    print("-" * 80)
    print_results("Rusty Machine L2 (GPU)", score_rm, duration_rm, "Accuracy", speedup=duration_sk / duration_rm if duration_rm > 0 else 0)
    print_results("Scikit-learn L2 (CPU)", score_sk, duration_sk, "Accuracy")
    print("-" * 80 + "\n")


# --- Benchmark 3: Logistic Regression (L1) ---

def run_logistic_l1_regression_benchmark():
    """
    Runs the benchmark for L1-regularized Logistic Regression (Lasso).
    
    Compares `RustyLogisticRegression` (GPU) against
    `sklearn.linear_model.LogisticRegression` (CPU 'saga' solver),
    as 'saga' is one of the few sklearn solvers to support L1.
    """
    print_header(f"BENCHMARK 3: LOGISTIC REGRESSION (L1 penalty, alpha={ALPHA})")

    # 1. Define dataset and model parameters
    SAMPLES = 500_000
    FEATURES = 100
    INFORMATIVE_FEATURES = 50
    EPOCHS = 100
    LR = 0.05
    BATCH_SIZE = 4096

    print(f"Generating data: {SAMPLES:,} samples, {FEATURES} features...\n")

    # 2. Generate, split, and scale data
    X, y = make_classification(
        n_samples=SAMPLES,
        n_features=FEATURES,
        n_informative=INFORMATIVE_FEATURES,
        n_redundant=10,
        random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Rusty Machine (GPU)
    print("Training Rusty Machine Logistic L1 (GPU)...")
    model_rm = RustyLogisticRegression(
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, penalty='l1', alpha=ALPHA, random_state=RANDOM_STATE
    )
    start_rm = time.time()
    model_rm.fit(X_train_scaled, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test_scaled)
    score_rm = accuracy_score(y_test, preds_rm)
    print(f"Training finished in {duration_rm:.4f} seconds.")

    # 4. Train Scikit-learn (CPU)
    print("\nTraining Scikit-learn Logistic L1 (CPU)...")
    # Convert alpha to sklearn's C (C = 1/alpha)
    c_param = 1.0 / ALPHA if ALPHA > 0 else float('inf')
    # 'saga' solver is required for L1 penalty
    model_sk = SklearnLogisticRegression(
        penalty='l1', C=c_param, solver='saga', max_iter=EPOCHS, tol=1e-3, random_state=RANDOM_STATE
    )
    start_sk = time.time()
    model_sk.fit(X_train_scaled, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test_scaled)
    score_sk = accuracy_score(y_test, preds_sk)
    print(f"Training finished in {duration_sk:.4f} seconds.")

    # 5. Print comparative results
    print("\n" + "-" * 80)
    print(f"{'Model':<30} | {'Metric':<15} | {'Accuracy':<10} | {'Train Time':<12} | {'Speedup':<10}")
    print("-" * 80)
    print_results("Rusty Machine L1 (GPU)", score_rm, duration_rm, "Accuracy", speedup=duration_sk / duration_rm if duration_rm > 0 else 0)
    print_results("Scikit-learn L1 (CPU)", score_sk, duration_sk, "Accuracy")
    print("-" * 80 + "\n")


# --- Main Execution ---

if __name__ == "__main__":
    """
    Runs all defined benchmarks in sequence.
    """
    run_linear_regression_benchmark()
    run_logistic_l2_regression_benchmark()
    run_logistic_l1_regression_benchmark()