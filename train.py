import os
import tensorflow as tf
from model import create_model
from utils import load_data

train_path = '/opt/ml/input/data/train'
val_path = '/opt/ml/input/data/val'
output_path = '/opt/ml/model'

print(">> Listing training input directory:")
print(os.listdir(train_path))

print(">> Listing validation input directory:")
print(os.listdir(val_path))

# Przykład podfolderów klas
print(">> Training classes preview:")
for name in os.listdir(train_path):
    class_path = os.path.join(train_path, name)
    if os.path.isdir(class_path):
        print(f"{name} -> {os.listdir(class_path)[:3]}")

print(">> Validation classes preview:")
for name in os.listdir(val_path):
    class_path = os.path.join(val_path, name)
    if os.path.isdir(class_path):
        print(f"{name} -> {os.listdir(class_path)[:3]}")

epochs = int(os.environ.get('EPOCHS', 10))
batch_size = int(os.environ.get('BATCH_SIZE', 32))

train_ds = load_data(train_path, batch_size=batch_size)
val_ds = load_data(val_path, batch_size=batch_size)

model = create_model(num_classes=7)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save(output_path)
