import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    weights = {i: total / (num_classes * max(1, c)) for i, c in enumerate(counts)}
    return weights

def upsample_to_balance(files, labels_np, class_names, target_multiple=1.0):
    counts = defaultdict(int)
    for lab in labels_np:
        counts[int(lab)] += 1
    max_count = max(counts.values())
    target = int(max_count * target_multiple)

    by_class = defaultdict(list)
    for f, lab in zip(files, labels_np):
        by_class[int(lab)].append(f)

    balanced_files, balanced_labels = [], []
    for cls_idx, lst in by_class.items():
        if len(lst) == 0:
            continue
        mul = math.ceil(target / len(lst))
        expanded = (lst * mul)[:target]
        random.shuffle(expanded)
        balanced_files.extend(expanded)
        balanced_labels.extend([cls_idx] * len(expanded))

    mix = list(zip(balanced_files, balanced_labels))
    random.shuffle(mix)
    balanced_files, balanced_labels = zip(*mix)
    return list(balanced_files), np.array(balanced_labels, dtype=np.int32)

def make_dataset(file_list, class_names, labels=None, img_size=(224,224), batch_size=32, shuffle=True):
    num_classes = len(class_names)
    if labels is None:
        labels = np.array([class_names.index(os.path.basename(os.path.dirname(f))) for f in file_list], dtype=np.int32)
    else:
        labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_list, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_list))

    def process_path(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, tf.one_hot(label, depth=num_classes)

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

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