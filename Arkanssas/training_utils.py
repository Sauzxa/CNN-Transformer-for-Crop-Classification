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
    "SparseCategoricalFocalLoss",
    "MacroF1Score",
    "CosineAnnealingLearningRate",
    "augment_timeseries",
    "make_augment_tf_wrapper",
]

import numpy as np
import tensorflow as tf
from tensorflow import keras


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


class SparseCategoricalFocalLoss(keras.losses.Loss):
    """Focal loss sparse (y entier) ; alpha scalaire optionnel (pondération globale)."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, name="sparse_categorical_focal", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return sparse_categorical_focal_loss(self.gamma, self.alpha)(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


class MacroF1Score(keras.metrics.Metric):
    """F1-macro : accumulation TP / FP / FN par classe sur la batch."""

    def __init__(self, n_classes: int, name: str = "macro_f1", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.n_classes = int(n_classes)
        self.tp = self.add_weight(shape=(self.n_classes,), name="tp", initializer="zeros")
        self.fp = self.add_weight(shape=(self.n_classes,), name="fp", initializer="zeros")
        self.fn = self.add_weight(shape=(self.n_classes,), name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        depth = self.n_classes
        y_oh = tf.one_hot(y_true, depth=depth, dtype=tf.float32)
        p_oh = tf.one_hot(pred, depth=depth, dtype=tf.float32)
        tp = tf.reduce_sum(y_oh * p_oh, axis=0)
        fp = tf.reduce_sum((1.0 - y_oh) * p_oh, axis=0)
        fn = tf.reduce_sum(y_oh * (1.0 - p_oh), axis=0)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        eps = tf.constant(1e-7, dtype=self.tp.dtype)
        prec = self.tp / (self.tp + self.fp + eps)
        rec = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * prec * rec / (prec + rec + eps)
        return tf.reduce_mean(f1)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_classes": self.n_classes})
        return cfg


class CosineAnnealingLearningRate(keras.callbacks.Callback):
    """lr(epoch) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * epoch / T_max))."""

    def __init__(
        self,
        lr_max: float = 1e-3,
        lr_min: float = 1e-6,
        t_max: int = 100,
        verbose: int = 0,
    ):
        super().__init__()
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.t_max = max(int(t_max), 1)
        self.verbose = int(verbose)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1.0 + float(np.cos(np.pi * float(epoch) / float(self.t_max)))
        )
        self.model.optimizer.learning_rate.assign(lr)
        if self.verbose and epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: lr={lr:.2e}")


def augment_timeseries(
    X: np.ndarray,
    mask: np.ndarray | None,
    *,
    p: float = 0.3,
    dropout_frac_min: float = 0.1,
    dropout_frac_max: float = 0.2,
    noise_std: float = 0.005,
    seed: int | None = None,
) -> np.ndarray:
    """
    Augmentation train uniquement : dropout temporel aléatoire (10–20 % des pas)
    mis à 0 + bruit gaussien sur bandes valides. `mask` n'est pas modifié (référence originale).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32).copy()
    n, t, c = X.shape
    for i in range(n):
        if rng.random() > p:
            continue
        frac = rng.uniform(dropout_frac_min, dropout_frac_max)
        k = max(1, int(round(t * frac)))
        drop_idx = rng.choice(t, size=k, replace=False)
        X[i, drop_idx, :] = 0.0
        noise = rng.normal(0.0, noise_std, size=(t, c)).astype(np.float32)
        valid = np.ones(t, dtype=bool) if mask is None else mask[i].astype(bool)
        for tt in range(t):
            if valid[tt]:
                X[i, tt, :] = X[i, tt, :] + noise[tt]
    return X


def make_augment_tf_wrapper(p: float = 0.3, seed: int = 42):
    """Retourne une fonction pour `tf.data.Dataset.map` (train seulement)."""

    def _aug(x, y):
        def _numpy_aug(xx):
            xx = np.asarray(xx, dtype=np.float32)
            m = np.any(np.abs(xx) > 1e-12, axis=-1)
            return augment_timeseries(xx[None, ...], m[None, ...], p=p, seed=seed)[0]

        x_aug = tf.numpy_function(_numpy_aug, [x], tf.float32)
        x_aug.set_shape(x.shape)
        return x_aug, y

    return _aug
