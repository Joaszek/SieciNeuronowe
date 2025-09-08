import tensorflow as tf
import os

def load_data(file_list, class_names, img_size=(224, 224), batch_size=32):
    """
    Creates a TensorFlow dataset from a list of image file paths.

    Parameters
    ----------
    file_list : list of str
        List of file paths to images.
    class_names : list of str
        List of class labels corresponding to the dataset.
    img_size : tuple of int, default=(224, 224)
        Target size (height, width) for resizing images.
    batch_size : int, default=32
        Number of samples per batch.

    Returns
    -------
    tf.data.Dataset
        A dataset yielding `(image, label)` pairs where:
        - `image` is a float32 tensor normalized to [0, 1] with shape `(img_size[0], img_size[1], 3)`
        - `label` is a one-hot encoded vector of length `num_classes`
    """
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