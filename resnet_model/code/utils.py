import tensorflow as tf
import os
import numpy as np


def make_dataset(file_list, class_names, labels=None, img_size=(224, 224), batch_size=32, shuffle=True):
    """
    Tworzy tf.data.Dataset dla obrazów.

    file_list : list[str]
        Lista ścieżek do obrazów.
    class_names : list[str]
        Nazwy klas (np. ['bcc','bkl','mel','nv']).
    labels : list[int] lub None
        Jeśli podane → używane bezpośrednio.
        Jeśli None → etykiety wyciągane z folderu w ścieżce pliku.
    """
    num_classes = len(class_names)

    if labels is None:
        # etykiety z katalogów
        labels = []
        for f in file_list:
            class_name = os.path.basename(os.path.dirname(f))
            labels.append(class_names.index(class_name))
        labels = np.array(labels, dtype=np.int32)
    else:
        labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img, tf.one_hot(label, depth=num_classes)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=True)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds