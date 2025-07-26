import os
import tensorflow as tf
from model import create_model
from utils import load_data

dataset_path = '/opt/ml/input/data/training'
output_path = '/opt/ml/model'

print(">> Listing training input directory:")
print(os.listdir(dataset_path))

# Jeżeli folder zawiera podfoldery klas:
print(">> Listing classes:")
for name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, name)
    if os.path.isdir(class_path):
        print(f"{name} -> {os.listdir(class_path)[:5]}")

epochs = int(os.environ.get('EPOCHS', 10))
batch_size = int(os.environ.get('BATCH_SIZE', 32))

train_ds = load_data(dataset_path, batch_size=batch_size, subset='training')
val_ds = load_data(dataset_path, batch_size=batch_size, subset='validation')

model = create_model(num_classes=7)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save(output_path)
