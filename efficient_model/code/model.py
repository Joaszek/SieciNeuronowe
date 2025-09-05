from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
import os
import tensorflow as tf

def _get_env(name, default=None):
    v = os.environ.get(f"SM_HP_{name}", os.environ.get(name, None))
    return default if v is None else v

def _get_float(name, default):
    v = _get_env(name, None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def create_model(num_classes=7, input_shape=(224, 224, 3)):
    l2w = _get_float("HEAD_L2", 1e-4)
    d1  = _get_float("HEAD_DROPOUT1", 0.2)
    d2  = _get_float("HEAD_DROPOUT2", 0.3)
    d3  = _get_float("HEAD_DROPOUT3", 0.3)

    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # odczepiamy w fazie 2 fine-tune

    reg = regularizers.l2(l2w)
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # ważne: training=False na zamrożonej bazie
    x = layers.GlobalAveragePooling2D()(x)

    # Block 1
    x = layers.Dropout(d1)(x)
    x = layers.Dense(512, kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(d2)(x)

    # Block 2
    x = layers.Dense(256, kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(d2)(x)

    # Residual MLP bottleneck: 256 -> 128 -> 256 i dodanie skipa
    skip = x
    r = layers.Dense(128, kernel_regularizer=reg, use_bias=False)(x)
    r = layers.BatchNormalization()(r)
    r = layers.Activation('swish')(r)
    r = layers.Dropout(d3)(r)
    r = layers.Dense(256, kernel_regularizer=reg, use_bias=False)(r)
    r = layers.BatchNormalization()(r)
    x = layers.Add()([skip, r])
    x = layers.Activation('swish')(x)
    x = layers.Dropout(d3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model
