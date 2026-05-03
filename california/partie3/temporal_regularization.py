from __future__ import annotations

import numpy as np


def build_valid_timestep_mask(X: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must be (N, T, C)")
    return np.any(np.abs(X) > eps, axis=-1)


def resample_temporal_linear(
    X: np.ndarray,
    n_steps: int,
    *,
    t_obs: np.ndarray | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must be (N, T_in, C)")
    n, t_in_len, c = X.shape
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if t_in_len == 1:
        return np.repeat(X, n_steps, axis=1)

    if t_obs is None:
        t_in = np.linspace(0.0, 1.0, t_in_len, dtype=np.float64)
    else:
        t_in = np.asarray(t_obs, dtype=np.float64).ravel()
        if t_in.shape[0] != t_in_len:
            raise ValueError("len(t_obs) must match T_in")

    t_out = np.linspace(float(t_in[0]), float(t_in[-1]), n_steps, dtype=np.float64)
    out = np.empty((n, n_steps, c), dtype=np.float32)
    for k, tk in enumerate(t_out):
        if tk <= t_in[0]:
            out[:, k, :] = X[:, 0, :]
            continue
        if tk >= t_in[-1]:
            out[:, k, :] = X[:, -1, :]
            continue
        i = int(np.searchsorted(t_in, tk, side="right") - 1)
        i = min(max(i, 0), t_in_len - 2)
        denom = float(t_in[i + 1] - t_in[i])
        alpha = 0.0 if denom < 1e-12 else (tk - float(t_in[i])) / denom
        out[:, k, :] = ((1.0 - alpha) * X[:, i, :] + alpha * X[:, i + 1, :]).astype(np.float32)
    return out


def resample_timestep_mask_nearest(
    valid: np.ndarray,
    n_steps: int,
    *,
    t_obs: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid must be (N, T_in)")
    n, t_in_len = valid.shape
    if t_in_len == 1:
        return np.broadcast_to(valid[:, :1].astype(np.float32), (n, n_steps)).copy()

    if t_obs is None:
        t_in = np.linspace(0.0, 1.0, t_in_len, dtype=np.float64)
    else:
        t_in = np.asarray(t_obs, dtype=np.float64).ravel()
        if len(t_in) != t_in_len:
            raise ValueError("len(t_obs) must match T_in")

    t_out = np.linspace(float(t_in[0]), float(t_in[-1]), n_steps, dtype=np.float64)
    out = np.zeros((n, n_steps), dtype=np.float32)
    for k, tk in enumerate(t_out):
        j = int(np.argmin(np.abs(t_in - tk)))
        out[:, k] = valid[:, j].astype(np.float32)
    return out


def append_observation_mask_channel(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[..., None]
    if X.shape[:2] != m.shape[:2]:
        raise ValueError("X and mask must share (N, T)")
    return np.concatenate([X, m], axis=-1)


__all__ = [
    "append_observation_mask_channel",
    "build_valid_timestep_mask",
    "resample_temporal_linear",
    "resample_timestep_mask_nearest",
]

