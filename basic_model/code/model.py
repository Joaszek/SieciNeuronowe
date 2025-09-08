from tensorflow.keras import layers, models


def create_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Builds and returns a convolutional neural network for image classification.

    Parameters
    ----------
    input_shape : tuple of int, default=(224, 224, 3)
        Shape of the input images (height, width, channels).
    num_classes : int, default=7
        Number of output classes for classification.

    Returns
    -------
    tensorflow.keras.Model
        A compiled Sequential model with convolutional, pooling, normalization,
        and dense layers designed for multi-class image classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
