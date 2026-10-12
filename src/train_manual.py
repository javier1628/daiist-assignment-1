"""Manual PyTorch linear regression. OWNER WRITES THIS FILE.

Contract:
    fit_manual(X: np.ndarray, y: np.ndarray, lr: float, l2: float, epochs: int) -> dict
        returns {"w": torch.Tensor[n_features], "b": torch.Tensor[1], "loss_curve": list[float]}
    predict_manual(params: dict, X: np.ndarray) -> np.ndarray

Rules of the game (from the Session 5 notebook):
  - no torch.nn, no torch.optim
  - w, b are tensors with requires_grad=True
  - loss = mean((X @ w + b - y)**2) + l2 * (w**2).sum()
  - loss.backward(); then inside torch.no_grad(): w -= lr * w.grad; b -= lr * b.grad
  - zero the grads every epoch
  - full batch, record loss every epoch so train.py can save the curve
"""
import numpy as np
import torch


def fit_manual(X: np.ndarray, y: np.ndarray, lr: float = 0.01, l2: float = 0.0, epochs: int = 2000) -> dict:
    # float64 to match sklearn's precision, so the three models can be compared coefficient
    # by coefficient. These tensors are plain data — they do NOT need gradients.
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    # The parameters we DO learn. Start at zero; requires_grad=True tells autograd to track them.
    w = torch.zeros(X_t.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    loss_curve: list[float] = []
    for _ in range(epochs):
        # Forward pass: linear prediction, then mean squared error + L2 penalty on w (not b).
        preds = X_t @ w + b
        loss = torch.mean((preds - y_t) ** 2) + l2 * (w ** 2).sum()

        # Backward pass: autograd fills w.grad and b.grad with d(loss)/d(param).
        loss.backward()

        # Manual gradient-descent step. no_grad() so this update isn't itself recorded.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # Grads accumulate by default, so clear them before the next epoch.
        w.grad.zero_()
        b.grad.zero_()

        loss_curve.append(float(loss.item()))

    # Detach so the saved params are pure data (no autograd graph hanging off them).
    return {"w": w.detach(), "b": b.detach(), "loss_curve": loss_curve}


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float64)
    with torch.no_grad():
        return (X_t @ params["w"] + params["b"]).numpy()
