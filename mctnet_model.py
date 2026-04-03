import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class ECA1D(layers.Layer):
    """
    Efficient Channel Attention for sequence (B, T, C): GAP over time -> 1D conv along
    channel axis -> sigmoid -> scale. Uses masked global average when mask is provided.
    """

    def __init__(self, kernel_size: int = 3, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            1,
            self.kernel_size,
            padding="same",
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_conv",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: x or (x, mask_bool)
        if isinstance(inputs, (list, tuple)):
            x, mask_bool = inputs
        else:
            x, mask_bool = inputs, None

        if mask_bool is not None:
            mask_f = tf.cast(mask_bool, x.dtype)[..., None]
            den = tf.reduce_sum(mask_f, axis=1) + 1e-6
            y = tf.reduce_sum(x * mask_f, axis=1) / den  # (B, C)
        else:
            y = tf.reduce_mean(x, axis=1)  # (B, C)

        y = y[:, :, None]  # (B, C, 1) — channels as length for Conv1D
        y = self.conv(y)  # (B, C, 1)
        y = tf.nn.sigmoid(tf.squeeze(y, axis=-1))  # (B, C)
        return x * y[:, None, :]


class ALPE(layers.Layer):
    """
    Attention-based Learnable Positional Encoding (paper):
    ALPE(t) = ECA(Conv1D(PE(t) * mask))
    """

    def __init__(
        self,
        num_timesteps: int,
        d_model: int,
        conv_kernel: int = 3,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_timesteps = int(num_timesteps)
        self.d_model = int(d_model)
        self.conv_kernel = int(conv_kernel)
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=True,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_conv",
        )
        self.eca = ECA1D(
            kernel_size=self.conv_kernel,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_eca",
        )
        super().build(input_shape)

    def call(self, mask_bool, **kwargs):
        # mask_bool: (B, T) True = valid
        # PE must be built with TF ops inside call() so Keras/tf.function tracing does not
        # capture a stale tf.constant from build() (cross-graph "out of scope" error on fit).
        dtype = self.compute_dtype or tf.float32
        T = self.num_timesteps
        d = self.d_model
        pos = tf.cast(tf.range(T)[:, None], dtype)
        i = tf.cast(tf.range(d // 2)[None, :], dtype)
        angle_rates = tf.pow(
            tf.constant(10000.0, dtype=dtype),
            -(2.0 * i) / tf.cast(d, dtype),
        )
        angles = pos * angle_rates
        sines = tf.sin(angles)
        cosines = tf.cos(angles)
        pe = tf.reshape(tf.concat([sines, cosines], axis=-1), [1, T, d])
        b = tf.shape(mask_bool)[0]
        pe = tf.tile(pe, [b, 1, 1])  # (B, T, d)
        m = tf.cast(mask_bool[..., None], pe.dtype)
        masked = pe * m
        h = self.conv(masked)
        return self.eca([h, mask_bool])


class MaskedGlobalAveragePooling1D(layers.Layer):
    """Global average pooling over the time dimension, ignoring masked timesteps."""

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        mask_f = tf.cast(mask, x.dtype)[..., None]
        x_sum = tf.reduce_sum(x * mask_f, axis=1)
        den = tf.reduce_sum(mask_f, axis=1)
        return x_sum / (den + self.eps)


class MaskedGlobalMaxPooling1D(layers.Layer):
    """Global max pooling over time; invalid steps ignored (set to large negative)."""

    def __init__(self, fill_value: float = -1e9, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = float(fill_value)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        mask_f = tf.cast(mask, x.dtype)[..., None]
        fill = tf.cast(self.fill_value, x.dtype)
        x_masked = tf.where(mask_f > 0, x, fill)
        return tf.reduce_max(x_masked, axis=1)


class EnsureOneValidTimeStep(layers.Layer):
    """Guarantee at least one valid timestep per sample for attention masks."""

    def call(self, m, **kwargs):
        m = tf.cast(m, tf.bool)
        any_valid = tf.reduce_any(m, axis=1, keepdims=True)
        b = tf.shape(m)[0]
        t = tf.shape(m)[1]
        fallback = tf.concat(
            [tf.ones((b, 1), dtype=tf.bool), tf.zeros((b, t - 1), dtype=tf.bool)],
            axis=1,
        )
        return tf.where(any_valid, m, fallback)

    def compute_output_shape(self, input_shape):
        return input_shape


class BuildSelfAttentionMask(layers.Layer):
    """Convert (B,T) valid mask -> (B,T,T) attention allow-mask."""

    def call(self, mask_bool, **kwargs):
        mask_bool = tf.cast(mask_bool, tf.bool)
        return tf.logical_and(mask_bool[:, :, None], mask_bool[:, None, :])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


class CTFusion(layers.Layer):
    """
    One CTFusion stage: local CNN branch (with ECA) + global Transformer branch
    on (LayerNorm(x + ALPE)); fused by sum + LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_timesteps: int,
        conv_kernel: int = 3,
        kernel_regularizer=None,
        name_prefix: str = "ct",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.num_timesteps = int(num_timesteps)
        self.conv_kernel = int(conv_kernel)
        self.kernel_regularizer = kernel_regularizer
        self.name_prefix = name_prefix
        self._key_dim = max(1, self.d_model // self.num_heads)

    def build(self, input_shape):
        p = self.name_prefix
        kr = self.kernel_regularizer
        self.alpe = ALPE(
            self.num_timesteps,
            self.d_model,
            conv_kernel=self.conv_kernel,
            kernel_regularizer=kr,
            name=f"{p}_alpe",
        )
        self.cnn_bn1 = layers.BatchNormalization(name=f"{p}_cnn_bn1")
        self.cnn_conv1 = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=False,
            kernel_regularizer=kr,
            name=f"{p}_cnn_conv1",
        )
        self.cnn_eca1 = ECA1D(
            kernel_size=self.conv_kernel,
            kernel_regularizer=kr,
            name=f"{p}_cnn_eca1",
        )
        self.cnn_bn2 = layers.BatchNormalization(name=f"{p}_cnn_bn2")
        self.cnn_conv2 = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=False,
            kernel_regularizer=kr,
            name=f"{p}_cnn_conv2",
        )
        self.cnn_eca2 = ECA1D(
            kernel_size=self.conv_kernel,
            kernel_regularizer=kr,
            name=f"{p}_cnn_eca2",
        )
        self.ln_in = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_in")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._key_dim,
            dropout=self.dropout,
            name=f"{p}_mha",
        )
        self.ln_attn = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_attn")
        self.ff1 = layers.Dense(
            self.ff_dim,
            activation="gelu",
            kernel_regularizer=kr,
            name=f"{p}_ff1",
        )
        self.ff_drop = layers.Dropout(self.dropout, name=f"{p}_ff_drop")
        self.ff2 = layers.Dense(self.d_model, kernel_regularizer=kr, name=f"{p}_ff2")
        self.ln_ff = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_ff")
        self.ln_fuse = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_fuse")
        self.relu = layers.Activation("relu", name=f"{p}_relu")
        self.add_cnn_res = layers.Add(name=f"{p}_cnn_res")
        self.add_pe = layers.Add(name=f"{p}_add_pe")
        self.add_attn_res = layers.Add(name=f"{p}_attn_res")
        self.add_ff_res = layers.Add(name=f"{p}_ff_res")
        self.add_fuse = layers.Add(name=f"{p}_fuse")
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, mask_bool, attn_mask = inputs

        alpe = self.alpe(mask_bool)

        # Local CNN branch (residual + ECA after first conv stack, paper-aligned kernel 3)
        res = x
        h = self.cnn_conv1(x)
        h = self.cnn_bn1(h, training=training)
        h = self.relu(h)
        h = self.cnn_eca1([h, mask_bool])
        h = self.cnn_conv2(h)
        h = self.cnn_bn2(h, training=training)
        h = self.relu(h)
        h = self.cnn_eca2([h, mask_bool])
        cnn_branch = self.add_cnn_res([res, h])

        # Global Transformer branch (pre-norm on x + ALPE)
        xt = self.ln_in(self.add_pe([x, alpe]))
        xa = self.mha(xt, xt, attention_mask=attn_mask, training=training)
        xt2 = self.ln_attn(self.add_attn_res([xt, xa]))
        xf = self.ff1(xt2)
        xf = self.ff_drop(xf, training=training)
        xf = self.ff2(xf)
        trans_branch = self.ln_ff(self.add_ff_res([xt2, xf]))

        return self.ln_fuse(self.add_fuse([cnn_branch, trans_branch]))


def build_mctnet(
    n_timesteps: int,
    n_channels: int,
    n_classes: int,
    n_static_features: int | None = None,
    *,
    d_model: int = 32,
    num_heads: int = 5,
    ff_dim: int = 64,
    n_stage: int | None = None,
    num_transformer_blocks: int | None = None,
    dropout: float = 0.1,
    missing_value: float | None = 0.0,
    add_ndvi_if_missing: bool = True,
    light: bool = True,
    conv_kernel: int = 3,
    l2: float | None = None,
) -> keras.Model:
    """
    MCTNet-style multi-stage CNN–Transformer (CTFusion × n_stage) with full ALPE
    (sinusoidal PE × mask → Conv1D → ECA), ECA in CNN branches, masked global **max**
    pooling at the output (paper §2.3.2).

    `num_transformer_blocks` is deprecated; use `n_stage` (default 3, paper Table 3).
    d_model must be even (sinusoidal PE). Prefer d_model divisible by num_heads for
    clean head splits (e.g. d_model=60, num_heads=5).
    Optional `l2` adds kernel L2 regularization on conv/dense layers (reduces overfitting).
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal ALPE, got {d_model}")

    if n_stage is None:
        n_stage = num_transformer_blocks if num_transformer_blocks is not None else 3
    else:
        if num_transformer_blocks is not None and num_transformer_blocks != n_stage:
            raise ValueError("Pass only one of n_stage and num_transformer_blocks")

    if light:
        d_model = min(d_model, 32)
        n_stage = min(n_stage, 3)
        num_heads = min(num_heads, 5)
        ff_dim = min(ff_dim, 64)

    kernel_reg = regularizers.l2(l2) if l2 is not None and float(l2) > 0 else None

    # Inputs
    inp = keras.Input(shape=(n_timesteps, n_channels), dtype=tf.float32, name="x_s2_clim")
    model_inputs = [inp]

    if missing_value is None:
        mask_bool = layers.Lambda(
            lambda t: tf.reduce_any(tf.math.is_finite(t), axis=-1),
            name="time_valid_mask",
        )(inp)
    else:
        mask_bool = layers.Lambda(
            lambda t: tf.reduce_any(
                tf.logical_and(
                    tf.math.is_finite(t),
                    tf.not_equal(t, tf.cast(missing_value, t.dtype)),
                ),
                axis=-1,
            ),
            name="time_valid_mask",
        )(inp)
    mask_bool = layers.Lambda(lambda m: tf.cast(m, tf.bool), name="time_valid_mask_bool")(mask_bool)
    mask_bool = EnsureOneValidTimeStep(name="ensure_one_valid")(mask_bool)
    mask_bool = layers.Lambda(lambda m: tf.cast(m, tf.bool), name="time_valid_mask_final_bool")(mask_bool)

    attn_mask = BuildSelfAttentionMask(name="self_attention_mask")(mask_bool)

    x_clean = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)),
        name="replace_non_finite_by_zero",
    )(inp)

    x_feat = x_clean
    if add_ndvi_if_missing and n_channels == 10:
        eps = tf.constant(1e-6, dtype=x_clean.dtype)
        b04 = x_clean[:, :, 2]
        b08 = x_clean[:, :, 3]
        ndvi = (b08 - b04) / (b08 + b04 + eps)
        ndvi = ndvi[:, :, None]
        x_feat = layers.Concatenate(axis=-1, name="append_ndvi")([x_clean, ndvi])

    # Project input channels → d_model
    x = layers.Conv1D(
        d_model,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=kernel_reg,
        name="stem_conv",
    )(x_feat)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    for s in range(n_stage):
        x = CTFusion(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_timesteps=n_timesteps,
            conv_kernel=conv_kernel,
            kernel_regularizer=kernel_reg,
            name_prefix=f"stage{s}",
            name=f"ct_fusion_{s}",
        )([x, mask_bool, attn_mask])

    x = MaskedGlobalMaxPooling1D(name="masked_gmp")([x, mask_bool])
    
    if n_static_features is not None and n_static_features > 0:
        inp_static = keras.Input(shape=(n_static_features,), dtype=tf.float32, name="x_static")
        model_inputs.append(inp_static)
        
        # Projection of static features
        x_static = layers.Dense(d_model, activation="relu", kernel_regularizer=kernel_reg, name="static_dense1")(inp_static)
        x_static = layers.BatchNormalization(name="static_bn")(x_static)
        
        # Concatenate temporal and static features
        x = layers.Concatenate(axis=-1, name="concat_temporal_static")([x, x_static])

    x = layers.Dropout(dropout, name="head_drop")(x)
    out = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=kernel_reg,
        name="cls",
    )(x)
    return keras.Model(inputs=model_inputs, outputs=out, name="mctnet_masked_cnn_transformer")
