import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mctnet_model import (
    CTFusion,
    BuildSelfAttentionMask,
    EnsureOneValidTimeStep,
    MaskedGlobalAveragePooling1D,
    MaskedGlobalMaxPooling1D,
)


class CrossModalSequenceAttention(layers.Layer):
    """
    Cross-attention sur séquences complètes : requêtes = S2 (T2 pas de temps),
    clés / valeurs = S1 (T1 pas de temps). Le softmax porte sur T1 (>1), donc
    l'attention n'est pas dégénérée (contrairement au pooling préalable).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        *,
        attn_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) doit être divisible par num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = max(1, d_model // num_heads)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=float(attn_dropout),
            name=f"{self.name}_mha",
        )
        self.ln = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln")

    def call(self, inputs, training=None):
        x2, mask_s2, x1, mask_s1 = inputs
        m2 = tf.cast(mask_s2, tf.bool)
        m1 = tf.cast(mask_s1, tf.bool)
        # (B, T2, T1) : position (i,j) valide ssi requête i et clé j valides
        attn_mask = tf.logical_and(m2[:, :, None], m1[:, None, :])
        out = self.mha(
            query=x2,
            value=x1,
            key=x1,
            attention_mask=attn_mask,
            training=training,
        )
        return self.ln(x2 + out)


def _masked_pool(
    x: tf.Tensor,
    mask_bool: tf.Tensor,
    *,
    mode: str,
    name_prefix: str,
) -> tf.Tensor:
    if mode == "avg":
        return MaskedGlobalAveragePooling1D(name=f"{name_prefix}_pool_avg")([x, mask_bool])
    if mode == "max":
        return MaskedGlobalMaxPooling1D(name=f"{name_prefix}_pool_max")([x, mask_bool])
    raise ValueError("mode doit être 'avg' ou 'max'")


def _se_block_time_series(x: tf.Tensor, d_model: int, ratio: int = 8, name_prefix: str = "s2_se") -> tf.Tensor:
    """Squeeze-and-Excitation sur la dimension canal (moyenne temporelle puis ré-échelle)."""
    gap = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    hidden = max(d_model // ratio, 4)
    excite = layers.Dense(hidden, activation="relu", name=f"{name_prefix}_dense1")(gap)
    excite = layers.Dense(d_model, activation="sigmoid", name=f"{name_prefix}_dense2")(excite)
    excite = layers.Reshape((1, d_model), name=f"{name_prefix}_reshape")(excite)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, excite])


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
    l2: float | None = 5e-4,
    use_s1_branch: bool = True,
    use_static_branch: bool = True,
    s2_use_se: bool = True,
    s2_post_gru: bool = True,
    temporal_pooling: str = "avg",
) -> keras.Model:
    """
    MCTNet-v2 multimodal.

    - Cross-attention S2←S1 **avant** pooling temporel (séquences T2×d et T1×d).
    - Fusion tardive : concaténation des deux embeddings poolés + projection Dense(d_model).
    - temporal_pooling : 'avg' (recommandé) ou 'max' pour les vecteurs finaux S2/S1.
    """
    if temporal_pooling not in ("avg", "max"):
        raise ValueError("temporal_pooling doit être 'avg' ou 'max'")
    kernel_reg = keras.regularizers.l2(l2) if l2 else None

    inp_s2 = keras.Input(shape=(n_timesteps_s2, n_channels_s2), name="s2_input")
    x2 = layers.Conv1D(d_model, 1, activation="relu", kernel_regularizer=kernel_reg, name="s2_stem")(inp_s2)
    x2 = layers.BatchNormalization(name="s2_stem_bn")(x2)
    if s2_use_se:
        x2 = _se_block_time_series(x2, d_model, name_prefix="s2_se")

    mask_s2 = layers.Lambda(lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1))(inp_s2)
    mask_s2 = EnsureOneValidTimeStep()(mask_s2)
    attn_mask_s2 = BuildSelfAttentionMask()(mask_s2)

    for s in range(n_stage):
        x2 = CTFusion(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_timesteps=n_timesteps_s2,
            name_prefix=f"s2_st{s}",
        )([x2, mask_s2, attn_mask_s2])

    if s2_post_gru:
        x2 = layers.GRU(d_model, return_sequences=True, dropout=dropout, name="s2_gru")(x2)
        x2 = layers.BatchNormalization(name="s2_gru_bn")(x2)

    model_inputs: list = [inp_s2]

    if use_s1_branch:
        inp_s1 = keras.Input(shape=(n_timesteps_s1, n_channels_s1), name="s1_input")
        model_inputs.append(inp_s1)

        x1 = layers.Conv1D(d_model, 1, activation="relu", kernel_regularizer=kernel_reg, name="s1_stem")(inp_s1)
        x1 = layers.BatchNormalization()(x1)
        mask_s1 = layers.Lambda(lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1))(inp_s1)
        mask_s1 = EnsureOneValidTimeStep()(mask_s1)
        attn_mask_s1 = BuildSelfAttentionMask()(mask_s1)
        for s in range(n_stage):
            x1 = CTFusion(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_timesteps=n_timesteps_s1,
                name_prefix=f"s1_st{s}",
            )([x1, mask_s1, attn_mask_s1])

        x2 = CrossModalSequenceAttention(
            d_model=d_model,
            num_heads=cross_modal_heads,
            attn_dropout=cross_attn_dropout,
            name="cross_modal_seq_attn",
        )([x2, mask_s2, x1, mask_s1])

        pool_s2 = _masked_pool(x2, mask_s2, mode=temporal_pooling, name_prefix="s2")
        pool_s1 = _masked_pool(x1, mask_s1, mode=temporal_pooling, name_prefix="s1")
        fused = layers.Concatenate(name="modal_concat")([pool_s2, pool_s1])
        fused = layers.Dense(d_model, activation="relu", kernel_regularizer=kernel_reg, name="modal_fuse_dense")(
            fused
        )
        fused = layers.BatchNormalization(name="modal_fuse_bn")(fused)
    else:
        fused = _masked_pool(x2, mask_s2, mode=temporal_pooling, name_prefix="s2")

    if use_static_branch and n_static_features is not None and n_static_features > 0:
        inp_static = keras.Input(shape=(n_static_features,), name="static_input")
        model_inputs.append(inp_static)
        x_static = layers.Dense(d_model, activation="relu", kernel_regularizer=kernel_reg, name="static_dense")(
            inp_static
        )
        x_static = layers.BatchNormalization()(x_static)
        fused = layers.Concatenate(name="final_fusion")([fused, x_static])
    elif use_static_branch:
        raise ValueError("use_static_branch=True mais n_static_features manquant ou 0")

    head_units = max(128, d_model * 2)
    x = layers.Dense(head_units, activation="relu", kernel_regularizer=kernel_reg)(fused)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=kernel_reg)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="prediction")(x)

    return keras.Model(inputs=model_inputs, outputs=out, name="mctnet_v2_multimodal")


if __name__ == "__main__":
    m = build_mctnet_v2(3, 11, 12, 3, 5, 10, use_s1_branch=True, use_static_branch=True)
    m.summary()
    m2 = build_mctnet_v2(3, 11, 12, 3, 5, None, use_s1_branch=False, use_static_branch=False)
    m2.summary()
