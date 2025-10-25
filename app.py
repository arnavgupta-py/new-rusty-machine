import streamlit as st
import numpy as np
import cupy as cp
import pandas as pd
import time
import altair as alt # Required for coefficient plots
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
# Use Ridge to match RustyLinearRegression's L2
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import accuracy_score, r2_score

# Import Rusty Machine models
try:
    from rustymachine_api.models import LogisticRegression as RustyLogisticRegression
    from rustymachine_api.models import LinearRegression as RustyLinearRegression
except ImportError:
    st.error("FATAL ERROR: 'rusty_machine' library not found.")
    st.info("Please ensure you have run 'maturin develop --release' in your environment.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Rusty Machine",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
<style>
    :root {
        --color-bg: #0E1117;
        --color-bg-card: #1C202B;
        --color-border: #444444;
        --color-text: #FAFAFA;
        --color-text-dim: #AAAAAA;
        --color-primary: #00A0B0;
        --color-success: #00C853;
        --color-fail: #FF5252;
        --font-main: 'sans-serif';
        --font-mono: 'monospace';
    }
    body {
        font-family: var(--font-main);
        color: var(--color-text);
    }
    .main .block-container {
        padding: 2rem 3rem;
    }
    .title {
        font-family: var(--font-mono);
        color: var(--color-text);
        text-align: center;
        padding: 1rem;
        font-size: 2.5rem;
    }
    h1, h2, h3 {
        font-family: var(--font-mono);
        color: var(--color-text);
        font-weight: 600;
    }
    h3 {
         border-bottom: 1px solid var(--color-border);
         padding-bottom: 0.5rem;
         margin-top: 1.5rem;
         margin-bottom: 1rem;
    }
    /* Metric Cards (Old Style for Reference) */
    .metric-card-old {
        background-color: var(--color-bg-card); /* Slightly lighter card bg */
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0; /* Vertical margin */
        border: 1px solid var(--color-border);
        text-align: center;
    }
    .metric-card-old h3 {
        color: var(--color-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: normal;
        border-bottom: none; /* Remove border from card titles */
    }
    .metric-card-old p {
        font-size: 2.5rem;
        color: var(--color-text);
        font-weight: bold;
        margin: 0;
    }

     /* Streamlit Metric Styling */
    [data-testid="stMetric"] {
        background-color: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 1.5rem;
    }
     [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-family: var(--font-mono);
        color: var(--color-primary);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: var(--color-success); /* Green for value */
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] {
         font-size: 1.0rem;
        font-family: var(--font-mono);
        color: var(--color-text-dim);
    }

    /* Sidebar */
     [data-testid="stSidebar"] {
        background-color: var(--color-bg-card);
        border-right: 1px solid var(--color-border);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def format_bytes(byte_count):
    if byte_count is None or byte_count == 0: return "0.00 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"

def train_rusty_model(model_type, X_train, y_train, epochs, lr, batch_size, alpha, penalty):
    try:
        if model_type == "Logistic Regression":
            model = RustyLogisticRegression(
                epochs=epochs, lr=lr, batch_size=batch_size,
                penalty=penalty, alpha=alpha, random_state=42
            )
        else: # Linear Regression (Ridge)
            model = RustyLinearRegression(alpha=alpha)

        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        # Estimate GPU memory
        gpu_mem_used = cp.asarray(X_np).nbytes + cp.asarray(y_np).nbytes

        start_time = time.time()
        model.fit(X_np, y_np)
        duration = time.time() - start_time

        return duration, model, gpu_mem_used
    except Exception as e:
        st.error(f"Error in Rusty Machine: {e}")
        return -1, None, 0

def train_sklearn_model(model_type, X_train, y_train, epochs, alpha, penalty):
    try:
        if model_type == "Logistic Regression":
            # Convert alpha to C for scikit-learn
            C_param = 1.0 / alpha if alpha > 0 else float('inf')
            model = SklearnLogisticRegression(
                penalty=penalty, C=C_param, solver='saga',
                max_iter=epochs, tol=1e-3, random_state=42
            )
        else: # Linear Regression (Ridge)
            model = SklearnRidge(alpha=alpha, solver='auto', random_state=42)

        cpu_mem_used = X_train.nbytes + y_train.nbytes

        start_time = time.time()
        model.fit(X_train, y_train.ravel())
        duration = time.time() - start_time

        return duration, model, cpu_mem_used
    except Exception as e:
        st.error(f"Error in Scikit-learn: {e}")
        return -1, None, 0

# --- Sidebar Configuration ---

with st.sidebar:
    st.header("Benchmark Configuration")

    model_type = st.selectbox(
        "Select Model Type",
        ("Logistic Regression", "Linear Regression"),
        key="model_type_select"
    )

    if model_type == "Logistic Regression":
        default_samples, default_features = 500000, 100
        penalty = st.radio("Regularization Type", ('l2', 'l1'), key='penalty_radio')
    else:
        default_samples, default_features = 1000000, 100
        penalty = 'l2' # Linear Regression only supports L2 (Ridge)

    st.markdown("---")
    st.subheader("Dataset Parameters")
    n_samples = st.slider(
        "Dataset Samples",
        min_value=10000, max_value=1000000, value=default_samples, step=10000,
        key="n_samples_slider"
    )

    n_features = st.slider(
        "Dataset Features",
        min_value=10, max_value=200, value=default_features, step=10,
        key="n_features_slider"
    )

    st.markdown("---")
    st.subheader("Model Parameters")
    alpha = st.slider(
        "Regularization Strength (alpha)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        key="alpha_slider"
    )

    if model_type == "Logistic Regression":
        epochs = st.slider("Epochs", 50, 500, 100, 10, key="epochs_slider") # Reduced default epochs
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.05, 0.001, format="%.3f", key="lr_slider")
        batch_size = st.select_slider("Batch Size", [256, 512, 1024, 2048, 4096], 1024, key="bs_slider") # Smaller default BS
    else:
        # Defaults for Linear Regression (not used by Ridge/Normal Eq)
        epochs, learning_rate, batch_size = 1, 0.01, 1

    st.markdown("---")
    run_button = st.button("Initiate Benchmark", use_container_width=True, type="primary")


# --- Main Page ---

st.markdown('<h1 class="title">Rusty Machine // Performance & Regularization</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #AAAAAA; border:none; margin-top: -1rem;'>GPU Acceleration vs CPU & Effects of Regularization</h3>", unsafe_allow_html=True)
st.markdown("---")


if not run_button:
    st.info("Configure the benchmark in the sidebar and click 'Initiate Benchmark'.")

if run_button:
    # --- Data Generation ---
    with st.spinner(f"Generating data for {model_type}..."):
        if model_type == "Logistic Regression":
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), noise=25, random_state=42)

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    st.success(f"Dataset generated: {n_samples:,} samples, {n_features} features.")
    st.markdown("---")

    # --- Model Training ---
    st.header("Benchmark in Progress...")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rusty Machine (GPU)")
        with st.spinner("Training..."):
            rusty_duration, rusty_model, gpu_mem = train_rusty_model(
                model_type, X_train_scaled, y_train, epochs, learning_rate, batch_size, alpha, penalty
            )
        st.success(f"Completed in {rusty_duration:.4f}s")

    with col2:
        st.subheader("Scikit-learn (CPU)")
        with st.spinner("Training..."):
            sklearn_duration, sklearn_model, cpu_mem = train_sklearn_model(
                model_type, X_train_scaled, y_train, epochs, alpha, penalty
            )
        st.success(f"Completed in {sklearn_duration:.4f}s")

    st.markdown("---")

    # --- Results Display ---
    st.header("Final Results")

    if rusty_model and sklearn_model:
        # --- Key Metrics ---
        speedup = sklearn_duration / rusty_duration if rusty_duration > 0 else float('inf')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rusty Machine Time (GPU)", f"{rusty_duration:.3f}s")
        with col2:
            st.metric("Scikit-learn Time (CPU)", f"{sklearn_duration:.3f}s", delta=f"{sklearn_duration - rusty_duration:.3f}s slower", delta_color="inverse")
        with col3:
            st.metric("Performance Gain", f"{speedup:.2f}x", delta="GPU vs CPU Speedup")

        st.markdown("---")

        # --- Model Performance Table ---
        st.subheader("Model Performance Comparison")
        if model_type == "Logistic Regression":
            rusty_preds = rusty_model.predict(X_test_scaled)
            sklearn_preds = sklearn_model.predict(X_test_scaled)
            rusty_score = accuracy_score(y_test, rusty_preds)
            sklearn_score = accuracy_score(y_test, sklearn_preds)
            metric_name = "Accuracy"
        else:
            rusty_preds = rusty_model.predict(X_test_scaled)
            sklearn_preds = sklearn_model.predict(X_test_scaled)
            rusty_score = r2_score(y_test, rusty_preds)
            sklearn_score = r2_score(y_test, sklearn_preds)
            metric_name = "R² Score"

        score_data = {
            'Model': ['Rusty Machine (GPU)', 'Scikit-learn (CPU)'],
            metric_name: [f"{rusty_score:.4f}", f"{sklearn_score:.4f}"]
        }
        score_df = pd.DataFrame(score_data).set_index('Model')
        st.dataframe(score_df)

        st.markdown("---")

        # --- Regularization Effects ---
        st.subheader(f"Regularization Effects (Alpha = {alpha:.2f})")
        
        try:
            rm_coefs = rusty_model.coef_
            sk_coefs = sklearn_model.coef_.flatten() # Ensure 1D

            if rm_coefs is not None and sk_coefs is not None:
                coef_df = pd.DataFrame({
                    'Feature Index': np.arange(len(rm_coefs)),
                    'Rusty Machine Coef': rm_coefs,
                    'Scikit-learn Coef': sk_coefs
                })
                
                coef_melt_df = pd.melt(coef_df, id_vars=['Feature Index'],
                                       value_vars=['Rusty Machine Coef', 'Scikit-learn Coef'],
                                       var_name='Model', value_name='Coefficient Value')

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Coefficient Magnitudes (Bar Chart)")
                    chart_coef_bar = alt.Chart(coef_melt_df).mark_bar(opacity=0.7).encode(
                        x=alt.X('Feature Index:O', axis=None), # Ordinal, no axis labels
                        y=alt.Y('Coefficient Value:Q'),
                        color=alt.Color('Model:N', scale=alt.Scale(range=['#00A0B0', '#00C853'])),
                        tooltip=['Feature Index', 'Model', 'Coefficient Value']
                    ).properties(
                        title=f'Coefficient Magnitudes ({penalty.upper()} Regularization)'
                    ).interactive()
                    st.altair_chart(chart_coef_bar, use_container_width=True)
                    
                with col2:
                    st.markdown("#### Coefficient Comparison (Scatter Plot)")
                    chart_coef_scatter = alt.Chart(coef_df).mark_point(opacity=0.7).encode(
                        x=alt.X('Rusty Machine Coef:Q', title='Rusty Machine Coefficient'),
                        y=alt.Y('Scikit-learn Coef:Q', title='Scikit-learn Coefficient'),
                        tooltip=['Feature Index', 'Rusty Machine Coef', 'Scikit-learn Coef']
                    ).properties(
                        title='Rusty Machine vs Scikit-learn Coefficients'
                    ).interactive()
                    st.altair_chart(chart_coef_scatter, use_container_width=True)

                if penalty == 'l1':
                    rm_zeros = np.sum(np.abs(rm_coefs) < 1e-6)
                    sk_zeros = np.sum(np.abs(sk_coefs) < 1e-6)
                    st.markdown(f"**Sparsity (L1):** Rusty Machine: **{rm_zeros}/{len(rm_coefs)}** zero coefficients. "
                                f"Scikit-learn: **{sk_zeros}/{len(sk_coefs)}** zero coefficients.")
                else:
                     st.markdown(f"**Shrinkage (L2):** Observe how coefficients are pushed towards zero compared to low alpha values.")

            else:
                 st.warning("Could not retrieve coefficients from one or both models.")
        
        except AttributeError:
             st.warning("Coefficients (`.coef_`) attribute not found on one or both models.")
        except Exception as e:
            st.error(f"Error plotting coefficients: {e}")


        st.markdown("---")

        # --- Other Tables (Memory, Predictions) ---
        col1, col2 = st.columns(2)
        with col1:
             st.markdown("### Resource Usage Comparison")
             mem_data = {
                 'Model': ['Rusty Machine (GPU)', 'Scikit-learn (CPU)'],
                 'Est. Data Memory': [format_bytes(gpu_mem), format_bytes(cpu_mem)]
             }
             mem_df = pd.DataFrame(mem_data).set_index('Model')
             st.dataframe(mem_df)

        with col2:
            st.markdown("### Prediction Comparison (First 5)")
            pred_data = {
                'Feat 1': X_test_scaled[:5, 0],
                'Feat 2': X_test_scaled[:5, min(1, n_features-1)], # Avoid index error if n_features=1
                'Feat 3': X_test_scaled[:5, min(2, n_features-1)],
                'Rusty': rusty_preds.flatten()[:5],
                'Sklearn': sklearn_preds.flatten()[:5],
            }
            pred_df = pd.DataFrame(pred_data).round(3)
            st.dataframe(pred_df)

    else:
        st.error("Benchmark failed for one or both models.")