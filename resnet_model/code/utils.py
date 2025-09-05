import tensorflow as tf
import os
import numpy as np

def make_dataset(file_list, class_names, labels=None, img_size=(224, 224), batch_size=32, shuffle=True):
    num_classes = len(class_names)

    if file_list is None or len(file_list) == 0:
        raise ValueError("make_dataset: pusta lista plików.")

    if labels is None:
        labels = []
        for f in file_list:
            class_name = os.path.basename(os.path.dirname(f))
            labels.append(class_names.index(class_name))
        labels = np.array(labels, dtype=np.int32)
    else:
        labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def process_path(path, label):
        img_bytes = tf.io.read_file(path)
        # decode_image obsłuży jpg/jpeg/png i ustawi channels=3
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, antialias=True)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, depth=num_classes)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=True)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
