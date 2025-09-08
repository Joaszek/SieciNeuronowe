from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
import os
import tensorflow as tf

def _get_env(name, default=None):
    """
    Retrieves an environment variable, checking both SageMaker-style ("SM_HP_*")
    and plain environment variables.

    Parameters
    ----------
    name : str
        Name of the environment variable (without the "SM_HP_" prefix).
    default : any, optional
        Default value to return if the variable is not found.

    Returns
    -------
    str or any
        The environment variable value if found, otherwise the default.
    """
    v = os.environ.get(f"SM_HP_{name}", os.environ.get(name, None))
    return default if v is None else v

def _get_float(name, default):
    """
    Retrieves an environment variable as a float.

    Parameters
    ----------
    name : str
        Name of the environment variable (without the "SM_HP_" prefix).
    default : float
        Default value to return if the variable is not found or conversion fails.

    Returns
    -------
    float
        The environment variable value converted to float, or the default.
    """
    v = _get_env(name, None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def create_model(num_classes=7, input_shape=(224, 224, 3)):
    """
    Builds an image classification model based on EfficientNetB0 with
    custom classification head and residual MLP bottleneck.

    The architecture includes:
    - A frozen EfficientNetB0 backbone (transfer learning).
    - Fully connected blocks with dropout, batch normalization, and swish activation.
    - A residual bottleneck block with skip connection.
    - Final dense softmax layer for multi-class classification.

    Parameters
    ----------
    num_classes : int, default=7
        Number of output classes for classification.
    input_shape : tuple of int, default=(224, 224, 3)
        Shape of input images (height, width, channels).

    Returns
    -------
    tensorflow.keras.Model
        A compiled Keras Model instance ready for training.
    """
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
