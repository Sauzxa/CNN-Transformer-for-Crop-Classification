"""
Helpers for imbalanced crop classification: balanced oversampling and a robust
EarlyStopping that compares val_loss as Python floats (avoids Keras 3 + TF edge
cases where restore_best_weights sticks to epoch 1).
"""

from __future__ import annotations

__all__ = [
    "oversample_balanced",
    "ScalarEarlyStopping",
    "sparse_categorical_crossentropy_label_smoothing",
    "sparse_categorical_focal_loss",
]

import numpy as np
import tensorflow as tf


def oversample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each class, draw with replacement until every class has the same count
    as the majority class. Shuffles the result.
    """
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    counts = {int(c): int(np.sum(y == c)) for c in classes}
    n_target = max(counts.values())
    if n_target <= 0:
        return X, y

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for c in classes:
        c = int(c)
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=n_target, replace=True)
        X_list.append(X[pick])
        y_list.append(y[pick])

    Xb = np.concatenate(X_list, axis=0)
    yb = np.concatenate(y_list, axis=0)
    perm = rng.permutation(len(yb))
    return Xb[perm].astype(X.dtype, copy=False), yb[perm].astype(y.dtype, copy=False)


def _to_float_scalar(x) -> float | None:
    if x is None:
        return None
    if tf.is_tensor(x):
        x = x.numpy()
    a = np.asarray(x).reshape(-1)
    if a.size == 0:
        return None
    return float(a[0])


class ScalarEarlyStopping(tf.keras.callbacks.Callback):
    """
    Same role as keras.callbacks.EarlyStopping, but compares the monitored value
    using plain Python floats (min/max + min_delta). Use when val_loss improves
    in logs but restore_best_weights incorrectly keeps epoch 1.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        mode: str = "min",
        patience: int = 20,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.restore_best_weights = restore_best_weights
        self.verbose = int(verbose)
        self.wait = 0
        self.best: float | None = None
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0

    def _is_better(self, cur: float, ref: float) -> bool:
        if self.mode == "min":
            return cur < ref - self.min_delta
        if self.mode == "max":
            return cur > ref + self.min_delta
        raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = _to_float_scalar(logs.get(self.monitor))
        if current is None:
            return

        if self.best is None:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        self.wait += 1
        if self._is_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(
                f"Epoch {self.stopped_epoch + 1}: early stopping "
                f"(ScalarEarlyStopping, monitor={self.monitor!r})"
            )
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose:
                print(
                    "Restoring model weights from the end of the best epoch: "
                    f"{self.best_epoch + 1} ({self.monitor}={self.best:.6f})."
                )
            self.model.set_weights(self.best_weights)


def sparse_categorical_crossentropy_label_smoothing(
    num_classes: int, smoothing: float = 0.08
):
    """
    Sparse labels + softmax predictions, with label smoothing (Keras < 3.4 may not
    support label_smoothing on SparseCategoricalCrossentropy).
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        n = tf.cast(int(num_classes), y_pred.dtype)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        one_hot = tf.one_hot(y_true, depth=int(num_classes), dtype=y_pred.dtype)
        smooth = (1.0 - smoothing) * one_hot + smoothing / n
        logp = tf.math.log(y_pred)
        return -tf.reduce_mean(tf.reduce_sum(smooth * logp, axis=-1))

    return loss


def sparse_categorical_focal_loss(gamma: float = 2.0, alpha: float | None = None):
    """
    Focal loss for integer sparse labels y_true (N, 1) or (N,) and y_pred softmax (N, C).
    Down-weights easy examples (helps heavy class imbalance).
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p = tf.gather_nd(y_pred, idx)
        ce = -tf.math.log(p)
        mod = tf.pow(1.0 - p, gamma)
        if alpha is not None:
            mod = mod * alpha
        return tf.reduce_mean(mod * ce)

    return loss
