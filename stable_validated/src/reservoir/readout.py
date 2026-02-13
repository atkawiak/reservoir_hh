"""
Readout module: feature extraction + train/evaluate classifiers & regressors.

Implements leakage-free protocols with scaler fit only on train set.
Supports ridge regression, logistic regression, and linear SVM.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReadoutResult:
    """Result of readout training and evaluation."""
    model_type: str
    best_hparam: float
    train_score: float
    val_score: float
    test_score: float
    metric_name: str
    predictions: Optional[np.ndarray] = None


# ─── Feature Extraction ───

def extract_features_firing_rate(firing_rates: np.ndarray, washout_steps: int,
                                  downsample: int = 1) -> np.ndarray:
    """
    Extract features from filtered firing rates.

    Args:
        firing_rates: (n_steps, N) filtered firing rates
        washout_steps: number of initial steps to discard
        downsample: downsampling factor

    Returns:
        features: (n_samples, N)
    """
    data = firing_rates[washout_steps:]
    if downsample > 1:
        data = data[::downsample]
    return data


def extract_features_voltage(voltage_traces: np.ndarray, washout_steps: int,
                              downsample: int = 1) -> np.ndarray:
    """Extract features from voltage traces (control only)."""
    data = voltage_traces[washout_steps:]
    if downsample > 1:
        data = data[::downsample]
    return data


def extract_features_mean_window(firing_rates: np.ndarray, washout_steps: int,
                                  window_steps: int, stride_steps: int) -> np.ndarray:
    """
    Extract mean firing rate in sliding windows.

    Args:
        firing_rates: (n_steps, N)
        washout_steps: initial steps to discard
        window_steps: window size in steps
        stride_steps: stride in steps

    Returns:
        features: (n_windows, N)
    """
    data = firing_rates[washout_steps:]
    n_steps, N = data.shape
    features = []
    for start in range(0, n_steps - window_steps + 1, stride_steps):
        window = data[start:start + window_steps]
        features.append(window.mean(axis=0))
    return np.array(features)


def extract_features_concat_k(firing_rates: np.ndarray, washout_steps: int,
                                k: int, downsample: int = 1) -> np.ndarray:
    """
    Concatenate k consecutive frames as features.

    Args:
        firing_rates: (n_steps, N)
        washout_steps: steps to discard
        k: number of frames to concatenate
        downsample: downsampling factor applied first

    Returns:
        features: (n_samples, N*k)
    """
    data = firing_rates[washout_steps:]
    if downsample > 1:
        data = data[::downsample]
    n_steps, N = data.shape
    if n_steps < k:
        raise ValueError(f"Not enough samples ({n_steps}) for concat_k={k}")
    features = []
    for i in range(k - 1, n_steps):
        frame = data[i - k + 1:i + 1].flatten()
        features.append(frame)
    return np.array(features)


def extract_symbol_features(firing_rates: np.ndarray, washout_steps: int,
                             steps_per_symbol: int, method: str = "last",
                             downsample: int = 1) -> np.ndarray:
    """
    Extract one feature vector per symbol (for benchmark tasks).

    Args:
        firing_rates: (n_steps_stored, N)
        washout_steps: steps to discard (original scale)
        steps_per_symbol: integration steps per input symbol (original scale)
        method: "last" (last timestep) or "mean" (average over symbol)
        downsample: downsampling factor used during simulation
    """
    w_steps = washout_steps // downsample
    s_steps = steps_per_symbol // downsample
    
    data = firing_rates[w_steps:]
    n_stored, N = data.shape
    n_symbols = n_stored // s_steps
    features = np.zeros((n_symbols, N))

    for s in range(n_symbols):
        start = s * s_steps
        end = start + s_steps
        # 'mean' is standard for spiking LSM to capture the firing rate across the symbol
        features[s] = data[start:end].mean(axis=0)

    return features


# ─── Data Splitting ───

def split_data(X: np.ndarray, y: np.ndarray, train_frac: float, val_frac: float,
               rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets (sequential, no shuffle for time-series).

    Args:
        X: features (n_samples, n_features)
        y: targets (n_samples,) or (n_samples, n_outputs)
        train_frac: fraction for training
        val_frac: fraction for validation
        rng: unused for sequential split, kept for API compatibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Scaler ───

class StandardScaler:
    """StandardScaler that fits only on training data (anti-leakage)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ─── Readout Models ───

def ridge_regression(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     alpha_range: np.ndarray) -> ReadoutResult:
    """
    Ridge regression with hyperparameter sweep on validation set.

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        alpha_range: array of regularization values to try

    Returns:
        ReadoutResult
    """
    best_alpha = alpha_range[0]
    best_val_score = -np.inf
    best_w = None

    n_features = X_train.shape[1]
    I = np.eye(n_features)

    for alpha in alpha_range:
        # Closed-form ridge: w = (X^T X + alpha I)^{-1} X^T y
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX + alpha * I, Xty)
        except np.linalg.LinAlgError:
            continue

        # Validation score (negative NRMSE = we want to maximize)
        y_pred_val = X_val @ w
        val_score = -_nrmse(y_val, y_pred_val)

        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha
            best_w = w

    if best_w is None:
        return ReadoutResult("ridge", best_alpha, 0.0, 0.0, 0.0, "nrmse")

    # Train/test scores
    y_pred_train = X_train @ best_w
    train_score = -_nrmse(y_train, y_pred_train)
    y_pred_test = X_test @ best_w
    test_score = -_nrmse(y_test, y_pred_test)

    return ReadoutResult(
        model_type="ridge",
        best_hparam=best_alpha,
        train_score=-train_score,  # report NRMSE (lower is better → positive)
        val_score=-best_val_score,
        test_score=-test_score,
        metric_name="nrmse",
        predictions=y_pred_test,
    )


def logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         C_range: np.ndarray) -> ReadoutResult:
    """
    Logistic regression with hyperparameter sweep.

    Simple implementation using gradient descent (avoids sklearn dependency).
    For binary classification.
    """
    best_C = C_range[0]
    best_val_acc = -1
    best_w = None
    best_b = 0.0

    for C in C_range:
        w, b = _train_logistic(X_train, y_train, C, max_iter=500, lr=0.1)
        y_pred_val = _predict_logistic(X_val, w, b)
        val_acc = np.mean(y_pred_val == y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_w = w
            best_b = b

    if best_w is None:
        return ReadoutResult("logreg", best_C, 0.0, 0.0, 0.0, "accuracy")

    y_pred_train = _predict_logistic(X_train, best_w, best_b)
    train_acc = np.mean(y_pred_train == y_train)
    y_pred_test = _predict_logistic(X_test, best_w, best_b)
    test_acc = np.mean(y_pred_test == y_test)

    return ReadoutResult(
        model_type="logreg",
        best_hparam=best_C,
        train_score=train_acc,
        val_score=best_val_acc,
        test_score=test_acc,
        metric_name="accuracy",
        predictions=y_pred_test,
    )


def linear_svm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               C_range: np.ndarray) -> ReadoutResult:
    """
    Linear SVM via hinge loss SGD.

    Simple implementation for binary classification.
    """
    best_C = C_range[0]
    best_val_acc = -1
    best_w = None
    best_b = 0.0

    for C in C_range:
        w, b = _train_linear_svm(X_train, y_train, C, max_iter=500, lr=0.01)
        y_pred_val = _predict_linear(X_val, w, b)
        val_acc = np.mean(y_pred_val == y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_w = w
            best_b = b

    if best_w is None:
        return ReadoutResult("linear_svm", best_C, 0.0, 0.0, 0.0, "accuracy")

    y_pred_train = _predict_linear(X_train, best_w, best_b)
    train_acc = np.mean(y_pred_train == y_train)
    y_pred_test = _predict_linear(X_test, best_w, best_b)
    test_acc = np.mean(y_pred_test == y_test)

    return ReadoutResult(
        model_type="linear_svm",
        best_hparam=best_C,
        train_score=train_acc,
        val_score=best_val_acc,
        test_score=test_acc,
        metric_name="accuracy",
        predictions=y_pred_test,
    )


# ─── Helper functions ───

def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Root Mean Squared Error."""
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    if var < 1e-12:
        return 0.0
    return float(np.sqrt(mse / var))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _train_logistic(X: np.ndarray, y: np.ndarray, C: float,
                     max_iter: int = 500, lr: float = 0.1
                     ) -> Tuple[np.ndarray, float]:
    """Train logistic regression with L2 regularization."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    reg = 1.0 / (C * n) if C > 0 else 0.0

    for _ in range(max_iter):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = X.T @ (p - y) / n + reg * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def _predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    p = _sigmoid(X @ w + b)
    return (p >= 0.5).astype(np.float64)


def _train_linear_svm(X: np.ndarray, y: np.ndarray, C: float,
                       max_iter: int = 500, lr: float = 0.01
                       ) -> Tuple[np.ndarray, float]:
    """Train linear SVM with hinge loss and L2 reg via SGD."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    # Convert labels from {0,1} to {-1,+1}
    y_svm = 2.0 * y - 1.0

    for epoch in range(max_iter):
        # Mini-batch or full gradient
        margins = y_svm * (X @ w + b)
        hinge_mask = margins < 1.0

        grad_w = w - C * (X.T @ (y_svm * hinge_mask)) / n
        grad_b = -C * np.mean(y_svm * hinge_mask)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def _predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    scores = X @ w + b
    return (scores >= 0).astype(np.float64)
