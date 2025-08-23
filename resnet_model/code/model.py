from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score


def create_resnet_model(num_classes=4, input_shape=(224, 224, 3)):
    base_model = ResNet50(include_top=False, input_shape=input_shape, weights="imagenet")
    base_model.trainable = False  # zamrażamy backbone

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model