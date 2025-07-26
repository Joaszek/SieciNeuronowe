import tensorflow as tf

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
