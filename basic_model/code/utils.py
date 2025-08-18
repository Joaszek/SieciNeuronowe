import tensorflow as tf
import os

#def load_data(data_dir, img_size=(224, 224), batch_size=32):
#    return tf.keras.utils.image_dataset_from_directory(
#        data_dir,
#        seed=42,
#        image_size=img_size,
#        batch_size=batch_size,
#        label_mode='categorical'
#    )

def load_data(file_list, class_names, img_size=(224, 224), batch_size=32):
    """Tworzy dataset z listy plików"""
    num_classes = len(class_names)
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    def process_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        # Label jako indeks w class_names
        label_str = tf.strings.split(path, os.sep)[-2]
        label_idx = tf.cast(tf.where(tf.equal(class_names, label_str))[0][0], tf.int32)
        label_one_hot = tf.one_hot(label_idx, num_classes)
        return img, label_one_hot

    ds = ds.map(process_path)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds