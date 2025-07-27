import tensorflow as tf

def load_data(data_dir, img_size=(224, 224), batch_size=32, subset='training'):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
