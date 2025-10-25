"""
rustymachine_api: The Python API for the Rusty Machine library.

This package serves as the high-level Python interface for the
GPU-accelerated machine learning models implemented in Rust. It exposes
the core model classes, making them accessible for import by end-users
(e.g., `from rustymachine_api import LinearRegression`).

The `__all__` variable defines the public API of this package.
"""

# Import the model classes from the models.py submodule
# This allows users to import directly from the package root.
from .models import LinearRegression, LogisticRegression

# Define the public API. This list specifies which names are
# imported when a user runs `from rustymachine_api import *`.
__all__ = [
    "LinearRegression",
    "LogisticRegression"
]