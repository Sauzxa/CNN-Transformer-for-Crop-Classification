from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mctnet_model_paper import CTFusion


class CrossModalSequenceAttention(layers.Layer):
    def __init__(self, d_model: int, num_heads: int = 4, attn_dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.mha = layers.MultiHeadAttention(
            num_heads=int(num_heads),
            key_dim=max(1, int(d_model) // int(num_heads)),
            dropout=float(attn_dropout),
        )
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        x2, mask_s2, x1, mask_s1 = inputs
        m2 = tf.cast(mask_s2, tf.bool)
        m1 = tf.cast(mask_s1, tf.bool)
        attn_mask = tf.logical_and(m2[:, :, None], m1[:, None, :])
        out = self.mha(query=x2, value=x1, key=x1, attention_mask=attn_mask, training=training)
        return self.ln(x2 + out)


def _masked_pool_avg(x: tf.Tensor, mask_bool: tf.Tensor) -> tf.Tensor:
    m = tf.cast(mask_bool[..., None], x.dtype)
    x_sum = tf.reduce_sum(x * m, axis=1)
    denom = tf.maximum(tf.reduce_sum(m, axis=1), tf.constant(1.0, dtype=x.dtype))
    return x_sum / denom


def _build_sequence_branch(
    x_in: tf.Tensor,
    *,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    n_stage: int,
    dropout: float,
    prefix: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    x = layers.Conv1D(d_model, 1, activation="relu", name=f"{prefix}_stem")(x_in)
    x = layers.BatchNormalization(name=f"{prefix}_stem_bn")(x)
    mask = layers.Lambda(lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1), name=f"{prefix}_mask")(x_in)
    for s in range(n_stage):
        x = CTFusion(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            use_alpe=(s == 0),
            name=f"{prefix}_ct{s}",
        )(x, mask)
    return x, mask


def build_mctnet_v2(
    n_timesteps_s2: int,
    n_channels_s2: int,
    n_timesteps_s1: int,
    n_channels_s1: int,
    n_classes: int,
    n_static_features: int | None = None,
    *,
    d_model: int = 64,
    num_heads: int = 8,
    ff_dim: int = 192,
    n_stage: int = 3,
    dropout: float = 0.3,
    cross_attn_dropout: float = 0.1,
    cross_modal_heads: int = 4,
    use_s1_branch: bool = True,
    use_static_branch: bool = True,
) -> keras.Model:
    inp_s2 = keras.Input(shape=(n_timesteps_s2, n_channels_s2), name="s2_input")
    x2, mask_s2 = _build_sequence_branch(
        inp_s2,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        n_stage=n_stage,
        dropout=dropout,
        prefix="s2",
    )
    model_inputs: list[tf.Tensor] = [inp_s2]

    if use_s1_branch:
        inp_s1 = keras.Input(shape=(n_timesteps_s1, n_channels_s1), name="s1_input")
        model_inputs.append(inp_s1)
        x1, mask_s1 = _build_sequence_branch(
            inp_s1,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            n_stage=n_stage,
            dropout=dropout,
            prefix="s1",
        )
        x2 = CrossModalSequenceAttention(
            d_model=d_model,
            num_heads=cross_modal_heads,
            attn_dropout=cross_attn_dropout,
            name="cross_modal_seq_attn",
        )([x2, mask_s2, x1, mask_s1])
        pool_s2 = layers.Lambda(lambda z: _masked_pool_avg(z[0], z[1]), name="pool_s2")([x2, mask_s2])
        pool_s1 = layers.Lambda(lambda z: _masked_pool_avg(z[0], z[1]), name="pool_s1")([x1, mask_s1])
        fused = layers.Concatenate(name="modal_concat")([pool_s2, pool_s1])
        fused = layers.Dense(d_model, activation="relu", name="modal_fuse_dense")(fused)
        fused = layers.BatchNormalization(name="modal_fuse_bn")(fused)
    else:
        fused = layers.Lambda(lambda z: _masked_pool_avg(z[0], z[1]), name="pool_s2_only")([x2, mask_s2])

    if use_static_branch and n_static_features is not None and n_static_features > 0:
        inp_static = keras.Input(shape=(n_static_features,), name="static_input")
        model_inputs.append(inp_static)
        x_static = layers.Dense(d_model, activation="relu", name="static_dense")(inp_static)
        x_static = layers.BatchNormalization(name="static_bn")(x_static)
        fused = layers.Concatenate(name="final_fusion")([fused, x_static])
    elif use_static_branch:
        raise ValueError("use_static_branch=True but n_static_features is missing.")

    x = layers.Dense(max(128, d_model * 2), activation="relu", name="head_dense1")(fused)
    x = layers.Dropout(dropout, name="head_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="head_dense2")(x)
    x = layers.Dropout(dropout * 0.5, name="head_dropout2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="prediction")(x)
    return keras.Model(inputs=model_inputs, outputs=out, name="mctnet_v2_multimodal")

