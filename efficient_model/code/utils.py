import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import f1_score


def _get_env(name, default=None):
    """
    Retrieve environment variable, preferring SageMaker hyperparameters (SM_HP_*).

    Parameters
    ----------
    name : str
        Name of the environment variable (without SM_HP_ prefix).
    default : any, optional
        Default value if the variable is not set (default is None).

    Returns
    -------
    str or any
        Environment variable value, or default if not found.
    """
    # Najpierw sprawdź hyperparam z SageMaker (SM_HP_*)
    v = os.environ.get(f"SM_HP_{name}", os.environ.get(name, None))
    return default if v is None else v

def _get_bool(name, default=True):
    """
    Retrieve a boolean environment variable.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : bool, optional
        Default value if variable is not set (default True).

    Returns
    -------
    bool
        Boolean value of the environment variable.
    """
    v = _get_env(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1","true","t","yes","y")

def _get_float(name, default):
    """
    Retrieve a float environment variable.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : float
        Default value if variable is not set or invalid.

    Returns
    -------
    float
        Float value of the environment variable.
    """
    v = _get_env(name, None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def make_dataset(file_list, class_names, labels=None, img_size=(224, 224), batch_size=32, shuffle=True):
    """
    Create a TensorFlow dataset from image paths with optional augmentations and MixUp.

    Supported environment flags:
        - USE_AUG (default 1)
        - USE_MIXUP (default 1)
        - MIXUP_PROB (default 0.2)
        - MIXUP_MIN_LAM (default 0.3)
        - MIXUP_MAX_LAM (default 0.7)

    Parameters
    ----------
    file_list : list of str
        Paths to image files.
    class_names : list of str
        List of class labels.
    labels : list or array, optional
        Precomputed integer labels. If None, labels are inferred from folder names.
    img_size : tuple of int, default=(224,224)
        Target image size.
    batch_size : int, default=32
        Batch size.
    shuffle : bool, default=True
        Whether to shuffle the dataset.

    Returns
    -------
    tf.data.Dataset
        A batched and prefetched dataset yielding (image, one-hot label) pairs.
    """
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

    use_aug = _get_bool("USE_AUG", True) and shuffle
    use_mix = _get_bool("USE_MIXUP", True) and shuffle
    mix_prob = _get_float("MIXUP_PROB", 0.2)
    mix_min  = _get_float("MIXUP_MIN_LAM", 0.3)
    mix_max  = _get_float("MIXUP_MAX_LAM", 0.7)

    ds = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def process_path(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, antialias=True)
        img = preprocess_input(tf.cast(img, tf.float32))
        return img, tf.one_hot(label, depth=num_classes)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=True)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # --- klasyczne augmentacje (tylko train) ---
    if use_aug:
        h, w = img_size
        def aug(img, y):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_saturation(img, 0.9, 1.1)
            crop_frac = tf.random.uniform([], 0.9, 1.0)
            ch, cw = tf.cast(crop_frac*h, tf.int32), tf.cast(crop_frac*w, tf.int32)
            img = tf.image.random_crop(img, size=[ch, cw, 3])
            img = tf.image.resize(img, img_size, antialias=True)
            return img, y
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    # --- MIXUP po batchu (tylko train) ---
    if use_mix:
        @tf.function
        def apply_mixup(imgs, ys):
            do = tf.less(tf.random.uniform([]), tf.constant(mix_prob, dtype=tf.float32))
            def _mix():
                imgs2 = tf.concat([imgs[1:], imgs[:1]], axis=0)
                ys2   = tf.concat([ys[1:],   ys[:1]],   axis=0)
                lam   = tf.random.uniform([], mix_min, mix_max)
                imgs_m = lam * imgs + (1.0 - lam) * imgs2
                ys_m   = lam * ys   + (1.0 - lam) * ys2
                return imgs_m, ys_m
            return tf.cond(do, _mix, lambda: (imgs, ys))
        ds = ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



def make_alpha(labels_np, num_classes):
    """
    Create class weighting vector (alpha) for Focal Loss or similar.

    Parameters
    ----------
    labels_np : np.ndarray
        Integer labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    tf.Tensor
        Class weights normalized to sum 1.
    """
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(1.0, counts)
    alpha = inv / inv.sum()
    return tf.constant(alpha, dtype=tf.float32)

def categorical_focal_loss(gamma=2.0, alpha=None, label_smoothing=0.05):
    """
    Focal Loss for multi-class classification with optional label smoothing.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter for Focal Loss.
    alpha : tf.Tensor or None
        Class weighting vector [C] or None.
    label_smoothing : float, default=0.05
        Label smoothing factor.

    Returns
    -------
    function
        Loss function suitable for Keras `model.compile`.
    """
    def loss(y_true, y_pred):
        if label_smoothing > 0.0:
            n = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = (1.0 - label_smoothing) * y_true + label_smoothing / n
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        if alpha is not None:
            ce = ce * alpha  # [C] broadcast
        weight = tf.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss

class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, path):
        super().__init__()
        self.val_ds = val_ds
        self.best = -1.0
        self.path = path
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_ds, verbose=0)
        y_true = np.concatenate([np.argmax(y, axis=1) for _, y in self.val_ds], axis=0)
        y_pred = np.argmax(preds, axis=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\n[MacroF1Checkpoint] macro F1: {f1:.4f} (best {self.best:.4f})")
        if f1 > self.best:
            self.best = f1
            self.model.save(self.path)
            print(f"[MacroF1Checkpoint] Saved best to {self.path}")