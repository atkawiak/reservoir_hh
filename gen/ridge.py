"""Ridge regression readout for reservoir computing benchmarks.

Provides train/predict with L2-regularized linear regression.
Cross-validates regularization strength α from a grid.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


DEFAULT_ALPHAS = list(np.logspace(-6, 4, 21))


def ridge_cv_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float] | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[Ridge, float]:
    """Fit ridge regression with cross-validated α selection.

    Parameters
    ----------
    X_train : ndarray, shape (K_train, N)
        Training state matrix.
    y_train : ndarray, shape (K_train,)
        Training targets.
    alphas : list of float
        Regularization strengths to try. Default: DEFAULT_ALPHAS.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        RNG seed for fold splitting.

    Returns
    -------
    model : Ridge
        Fitted sklearn Ridge model with best α.
    best_alpha : float
        Selected regularization strength.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    best_score = -np.inf
    best_alpha = alphas[0]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for alpha in alphas:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train[train_idx], y_train[train_idx])
            score = model.score(X_train[val_idx], y_train[val_idx])
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    # Refit on full training set with best α
    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    return model, best_alpha


def ridge_predict(model: Ridge, X: np.ndarray) -> np.ndarray:
    """Predict with fitted ridge model."""
    return model.predict(X)
