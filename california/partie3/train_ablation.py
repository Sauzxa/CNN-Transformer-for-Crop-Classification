from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from partie3.mctnet_v2_model import build_mctnet_v2


def keras_class_weight_dict(y: np.ndarray, *, minority_boost: float = 1.2) -> dict[int, float]:
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    out = {int(c): float(wi) for c, wi in zip(classes, w)}
    if minority_boost and abs(minority_boost - 1.0) > 1e-6:
        counts = {int(c): int(np.sum(y == c)) for c in classes}
        med = float(np.median([counts[int(c)] for c in classes]))
        for c in classes:
            ci = int(c)
            if counts[ci] < med:
                out[ci] *= float(minority_boost)
    return out


def _slice_inputs(
    X_s2: np.ndarray,
    M_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    idx: np.ndarray,
    *,
    use_s1: bool,
    use_static: bool,
) -> list[np.ndarray]:
    out = [X_s2[idx], M_s2[idx]]
    if use_s1:
        out.append(X_s1[idx])
    if use_static:
        out.append(X_static[idx])
    return out


def _build_inputs_for_model(
    X_s2: np.ndarray,
    M_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    idx: np.ndarray,
    *,
    use_s1: bool,
    use_static: bool,
) -> list[np.ndarray]:
    s2 = X_s2[idx]
    m = M_s2[idx]
    if use_s1 and use_static:
        return [s2, X_s1[idx], X_static[idx]]
    if use_s1:
        return [s2, X_s1[idx]]
    if use_static:
        return [s2, X_static[idx]]
    return [s2]


def run_ablation_suite(
    X_s2: np.ndarray,
    M_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    n_classes: int = 6,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    minority_boost: float = 1.2,
    early_stopping_patience: int = 8,
    random_seed: int = 42,
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(random_seed)
    np.random.seed(random_seed)
    y_full = np.asarray(y).reshape(-1).astype(np.int32)
    cw = keras_class_weight_dict(y_full[idx_train], minority_boost=minority_boost)
    n_static = int(X_static.shape[1])

    configs: list[tuple[str, bool, bool]] = [
        ("S2_only", False, False),
        ("S2_static", False, True),
        ("S2_S1", True, False),
        ("S2_S1_static", True, True),
    ]
    ablation: dict[str, Any] = {}

    for name, use_s1, use_static in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_seed)
        n_st = n_static if use_static else None
        model = build_mctnet_v2(
            n_timesteps_s2=X_s2.shape[1],
            n_channels_s2=X_s2.shape[2],
            n_timesteps_s1=X_s1.shape[1],
            n_channels_s1=X_s1.shape[2],
            n_classes=n_classes,
            n_static_features=n_st,
            use_s1_branch=use_s1,
            use_static_branch=use_static,
            dropout=0.3,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")],
        )
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(early_stopping_patience),
                restore_best_weights=True,
                verbose=1,
            ),
        ]
        X_tr = _build_inputs_for_model(X_s2, M_s2, X_s1, X_static, idx_train, use_s1=use_s1, use_static=use_static)
        X_va = _build_inputs_for_model(X_s2, M_s2, X_s1, X_static, idx_val, use_s1=use_s1, use_static=use_static)
        X_te = _build_inputs_for_model(X_s2, M_s2, X_s1, X_static, idx_test, use_s1=use_s1, use_static=use_static)

        hist = model.fit(
            X_tr,
            y_full[idx_train],
            validation_data=(X_va, y_full[idx_val]),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=callbacks,
            verbose=1,
        )
        test_loss, test_acc, test_top2 = model.evaluate(X_te, y_full[idx_test], verbose=0)
        ablation[name] = {
            "history": hist,
            "model": model,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_top2": float(test_top2),
        }
    return ablation


__all__ = ["keras_class_weight_dict", "run_ablation_suite"]

