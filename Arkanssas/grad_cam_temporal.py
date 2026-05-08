"""
Grad-CAM / gradients sur l'entrée temporelle : importance (T, C) par pas de temps et canal.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def grad_cam_temporal(
    model: keras.Model,
    X_sample: np.ndarray,
    *,
    class_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradients de la log-probabilité de la classe cible par rapport aux entrées `x_s2_clim`.
    Retourne (heatmap (T, C), probas (n_classes,)).
    """
    if X_sample.ndim == 2:
        X_sample = X_sample[np.newaxis, ...]
    x = tf.cast(X_sample, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]).numpy())
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, x)[0]  # (T, C)
    heat = tf.abs(grads).numpy().astype(np.float32)
    m = heat.max()
    if m > 1e-9:
        heat = heat / m
    return heat, preds[0].numpy()


def plot_grad_cam_examples(
    model: keras.Model,
    X_per_class: dict[int, np.ndarray],
    *,
    save_path: str | None = None,
):
    """Affiche une heatmap (T, C) par classe."""
    import matplotlib.pyplot as plt

    n = max(1, len(X_per_class))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (cls, xv) in zip(axes[0], X_per_class.items()):
        hm, _ = grad_cam_temporal(model, xv, class_idx=cls)
        ax.imshow(hm.T, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_title(f"Classe {cls}")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Canal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
