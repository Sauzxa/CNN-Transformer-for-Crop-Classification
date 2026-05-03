import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class ECA1D(layers.Layer):
    def __init__(self, kernel_size=3, kernel_regularizer=None, **kwargs):
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
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, C)
        y = tf.reduce_mean(x, axis=1)  # (B, C)
        y = y[:, :, None]              # (B, C, 1)
        y = self.conv(y)               # (B, C, 1)
        y = tf.nn.sigmoid(tf.squeeze(y, axis=-1))  # (B, C)
        return x * y[:, None, :]


class SinusoidalPE(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)

    def call(self, x):
        # x: (B, T, C=d_model)
        dtype = x.dtype
        t = tf.shape(x)[1]
        pos = tf.cast(tf.range(t)[:, None], dtype)
        i = tf.cast(tf.range(self.d_model // 2)[None, :], dtype)
        angle_rates = tf.pow(tf.constant(10000.0, dtype=dtype), -(2.0 * i) / tf.cast(self.d_model, dtype))
        angles = pos * angle_rates
        pe = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  # (T, d_model)
        pe = pe[None, :, :]  # (1, T, d_model)
        return x + pe


class ALPE(layers.Layer):
    """
    ALPE(t) = ECA(Conv1D(PE(t) * mask))
    mask used only in first stage (paper).
    """
    def __init__(self, d_model, conv_kernel=3, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.conv_kernel = int(conv_kernel)
        self.kernel_regularizer = kernel_regularizer
        self.eca = ECA1D(kernel_size=conv_kernel, kernel_regularizer=kernel_regularizer)

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=True,
            kernel_regularizer=self.kernel_regularizer,
        )
        super().build(input_shape)

    def call(self, x, mask_bool):
        # x: (B, T, d_model), mask_bool: (B, T)
        dtype = x.dtype
        t = tf.shape(x)[1]
        pos = tf.cast(tf.range(t)[:, None], dtype)
        i = tf.cast(tf.range(self.d_model // 2)[None, :], dtype)
        angle_rates = tf.pow(tf.constant(10000.0, dtype=dtype), -(2.0 * i) / tf.cast(self.d_model, dtype))
        angles = pos * angle_rates
        pe = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)[None, :, :]  # (1, T, d_model)
        b = tf.shape(x)[0]
        pe = tf.tile(pe, [b, 1, 1])

        m = tf.cast(mask_bool[..., None], dtype)
        h = self.conv(pe * m)
        h = self.eca(h)
        return x + h


class TransformerSubModule(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout=0.1,
        conv_kernel=3,
        use_alpe=False,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)
        self.use_alpe = bool(use_alpe)
        self.kernel_regularizer = kernel_regularizer
        self.conv_kernel = int(conv_kernel)
        self.key_dim = max(1, self.d_model // self.num_heads)

    def build(self, input_shape):
        if self.use_alpe:
            self.pos = ALPE(
                d_model=self.d_model,
                conv_kernel=self.conv_kernel,
                kernel_regularizer=self.kernel_regularizer,
            )
        else:
            self.pos = SinusoidalPE(d_model=self.d_model)

        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout,
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = layers.Dense(
            self.ff_dim,
            activation="gelu",
            kernel_regularizer=self.kernel_regularizer,
        )
        self.ff_drop = layers.Dropout(self.dropout)
        self.ff2 = layers.Dense(self.d_model, kernel_regularizer=self.kernel_regularizer)
        super().build(input_shape)

    def call(self, x, mask_bool, training=None):
        # x: (B, T, d_model), mask_bool: (B, T)
        mask_bool = tf.cast(mask_bool, tf.bool)
        if self.use_alpe:
            h = self.pos(x, mask_bool)
        else:
            h = self.pos(x)

        attn_mask = tf.logical_and(mask_bool[:, :, None], mask_bool[:, None, :])  # (B,T,T)
        a = self.mha(h, h, attention_mask=attn_mask, training=training)
        h = self.ln1(h + a)
        f = self.ff1(h)
        f = self.ff_drop(f, training=training)
        f = self.ff2(f)
        return self.ln2(h + f)


class CNNSubModule(layers.Layer):
    def __init__(self, d_model, conv_kernel=3, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.conv_kernel = int(conv_kernel)
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(
            self.d_model,
            self.conv_kernel,
            padding="same",
            use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.Activation("relu")
        super().build(input_shape)

    def call(self, x, training=None):
        res = x
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.relu(h)
        return res + h


class CTFusion(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout=0.1,
        conv_kernel=3,
        use_alpe=False,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cnn = CNNSubModule(
            d_model=d_model,
            conv_kernel=conv_kernel,
            kernel_regularizer=kernel_regularizer,
        )
        self.trans = TransformerSubModule(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            conv_kernel=conv_kernel,
            use_alpe=use_alpe,
            kernel_regularizer=kernel_regularizer,
        )
        self.fuse_ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask_bool, training=None):
        x_cnn = self.cnn(x, training=training)
        x_tr = self.trans(x, mask_bool, training=training)
        return self.fuse_ln(x_cnn + x_tr)


def _mask_from_data(x, missing_value=0.0):
    if missing_value is None:
        return tf.reduce_any(tf.math.is_finite(x), axis=-1)
    return tf.reduce_any(
        tf.logical_and(
            tf.math.is_finite(x),
            tf.not_equal(x, tf.cast(missing_value, x.dtype)),
        ),
        axis=-1,
    )


def build_mctnet_paper(
    n_timesteps,
    n_channels,
    n_classes,
    n_static_features=None,
    *,
    d_model=20,
    num_heads=5,
    ff_dim=128,
    n_stage=3,
    conv_kernel=3,
    dropout=0.1,
    l2=0.0,
    missing_value=0.0,
):
    """
    Paper-oriented MCTNet:
    - Two inputs: spectral sequence + missing mask (Input1, Input2)
    - ALPE only in first Transformer sub-module
    - CTFusion stages with pooling between stages
    - Global max pooling + MLP classifier
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    kr = regularizers.l2(l2) if l2 and l2 > 0 else None

    x_in = keras.Input(shape=(n_timesteps, n_channels), dtype=tf.float32, name="x_s2")
    m_in = keras.Input(shape=(n_timesteps,), dtype=tf.float32, name="x_missing_mask")

    # if mask input is all zeros/ones wrong shape, fallback from data
    m_bool = layers.Lambda(lambda m: tf.cast(m > 0.5, tf.bool), name="mask_from_input")(m_in)
    m_data = layers.Lambda(lambda x: tf.cast(_mask_from_data(x, missing_value), tf.bool), name="mask_from_data")(x_in)
    m_bool = layers.Lambda(
        lambda z: tf.where(tf.reduce_any(z[0], axis=1, keepdims=True), z[0], z[1]),
        output_shape=lambda shapes: shapes[0],
        name="mask_safe",
    )([m_bool, m_data])

    x = layers.Lambda(
        lambda t: tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)),
        name="replace_non_finite",
    )(x_in)

    # stem projection
    x = layers.Conv1D(d_model, 1, padding="same", use_bias=False, kernel_regularizer=kr, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    channels = d_model
    for s in range(n_stage):
        if s > 0:
            channels = d_model * (2 ** s)
            x = layers.Conv1D(
                channels,
                1,
                padding="same",
                use_bias=False,
                kernel_regularizer=kr,
                name=f"stage{s}_proj",
            )(x)
            x = layers.BatchNormalization(name=f"stage{s}_proj_bn")(x)
            x = layers.Activation("relu", name=f"stage{s}_proj_relu")(x)

        x = CTFusion(
            d_model=channels,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            conv_kernel=conv_kernel,
            use_alpe=(s == 0),  # ALPE only first stage
            kernel_regularizer=kr,
            name=f"ct_fusion_{s}",
        )(x, m_bool)

        # pooling between stages (except last)
        if s < n_stage - 1:
            x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"stage{s}_pool")(x)
            m_pool = layers.Lambda(lambda m: tf.cast(m[..., None], tf.float32), name=f"stage{s}_mask_expand")(m_bool)
            m_pool = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name=f"stage{s}_mask_pool")(m_pool)
            m_bool = layers.Lambda(
                lambda m: tf.cast(tf.squeeze(m, axis=-1) > 0.5, tf.bool),
                name=f"stage{s}_mask_restore",
            )(m_pool)

    # masked global max pooling
    x_masked = layers.Lambda(
        lambda z: tf.where(tf.cast(z[1][..., None], tf.bool), z[0], tf.cast(-1e9, z[0].dtype)),
        output_shape=lambda shapes: shapes[0],
        name="mask_for_gmp",
    )([x, m_bool])
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x_masked)
    
    if n_static_features is not None and n_static_features > 0:
        inp_static = keras.Input(shape=(n_static_features,), dtype=tf.float32, name="x_static")
        model_inputs = [x_in, m_in, inp_static]
        x_static = layers.Dense(channels, activation="relu", kernel_regularizer=kr, name="static_dense1")(inp_static)
        x_static = layers.BatchNormalization(name="static_bn")(x_static)
        x = layers.Concatenate(axis=-1, name="concat_temporal_static")([x, x_static])
    else:
        model_inputs = [x_in, m_in]
        
    x = layers.Dropout(dropout, name="head_dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=kr, name="cls")(x)

    return keras.Model(inputs=model_inputs, outputs=out, name="mctnet_paper")

