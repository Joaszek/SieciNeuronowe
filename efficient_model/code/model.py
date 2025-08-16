from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def create_model(num_classes=7, input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # zamrażamy warstwy bazowe

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
