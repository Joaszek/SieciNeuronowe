from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

import os, shutil
import numpy as np
from collections import Counter
from glob import glob
from sklearn.model_selection import StratifiedKFold
from utils import upsample_to_balance, make_dataset, compute_class_weights, MacroF1Checkpoint
from model import create_resnet_model
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

TRAIN_CHANNEL = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

def train_kfold(all_files, class_names, num_classes=4, epochs=10, batch_size=32, num_folds=5):
    labels = np.array([class_names.index(os.path.basename(os.path.dirname(f))) for f in all_files])
    best_acc = 0.0
    best_model_path = '/opt/ml/model/best_model'
    fold_results = []

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(all_files, labels)):
        print(f"\n>>> Fold {i+1}/{num_folds}")
        train_files = [all_files[idx] for idx in train_idx]
        val_files = [all_files[idx] for idx in val_idx]

        train_files_bal, train_labels_bal = upsample_to_balance(train_files, labels[train_idx], class_names, target_multiple=0.5)

        train_ds = make_dataset(train_files_bal, class_names, labels=train_labels_bal, batch_size=batch_size, shuffle=True)
        val_ds = make_dataset(val_files, class_names, labels=labels[val_idx], batch_size=batch_size, shuffle=False)

        class_weights = compute_class_weights(labels[train_idx], num_classes)

        model = create_resnet_model(num_classes=num_classes)
        ckpt_dir = f'/opt/ml/model/fold_{i+1}'
        os.makedirs(ckpt_dir, exist_ok=True)

        callbacks = [
            MacroF1Checkpoint(val_ds, os.path.join(ckpt_dir, "best_f1.h5")),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1, cooldown=1),
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ]

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=callbacks, class_weight=class_weights)

        val_preds = model.predict(val_ds, verbose=0)
        val_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds], axis=0)
        val_pred_cls = np.argmax(val_preds, axis=1)

        acc = np.mean(val_pred_cls == val_true)
        print(f'Fold {i+1} Accuracy: {acc:.4f}')

        cm = confusion_matrix(val_true, val_pred_cls)
        print(f'Fold {i+1} Confusion Matrix:\n{cm}')

        report = classification_report(val_true, val_pred_cls, target_names=class_names)
        print(f'Classification Report:\n{report}')

        prec = precision_score(val_true, val_pred_cls, average='macro', zero_division=0)
        rec = recall_score(val_true, val_pred_cls, average='macro', zero_division=0)
        f1 = f1_score(val_true, val_pred_cls, average='macro', zero_division=0)
        fold_results.append((acc, prec, rec, f1))

        if acc > best_acc:
            best_acc = acc
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            model.save(best_model_path)
            print(f'>>> New best model saved: {best_model_path} (acc={best_acc:.4f})')

    print("\n===== TRAINING SUMMARY =====")
    for i, (acc, prec, rec, f1) in enumerate(fold_results, 1):
        print(f"Fold {i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    avg_acc = np.mean([r[0] for r in fold_results])
    avg_prec = np.mean([r[1] for r in fold_results])
    avg_rec = np.mean([r[2] for r in fold_results])
    avg_f1 = np.mean([r[3] for r in fold_results])
    print(f"\nAVG RESULTS: acc={avg_acc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}, f1={avg_f1:.4f}")

# ======= MAIN =======
exts = ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG")
all_files_raw = []
for e in exts:
    all_files_raw.extend(glob(os.path.join(TRAIN_CHANNEL, "**", e), recursive=True))

if len(all_files_raw) == 0:
    print(">>> DEBUG listing of TRAIN channel root:", TRAIN_CHANNEL)
    for root, dirs, files in os.walk(TRAIN_CHANNEL):
        print(root, "->", files[:5])
    raise ValueError(f"Nie znaleziono żadnych obrazów w {TRAIN_CHANNEL}. Upewnij się, że struktura to train/<klasa>/*.jpg")

labels_all = [os.path.basename(os.path.dirname(f)) for f in all_files_raw]
counts_all = Counter(labels_all)

# wybieramy top-4 najliczniejsze klasy
topk = min(4, len(counts_all))
top_classes = [cls for cls, _ in counts_all.most_common(topk)]

# pliki tylko do tych klas
all_files = [f for f in all_files_raw if os.path.basename(os.path.dirname(f)) in top_classes]
class_names = sorted(top_classes)
print("Final class names:", class_names)

# rozkład po filtracji (dla doboru liczby foldów)
post_counts = Counter([os.path.basename(os.path.dirname(f)) for f in all_files])
min_class_count = min(post_counts.values())
# K nie może być większe niż najmniejsza klasa i co najmniej 2
num_folds = max(2, min(5, min_class_count))

epochs = int(os.environ.get('EPOCHS', 10))
batch_size = int(os.environ.get('BATCH_SIZE', 32))

train_kfold(
    all_files,
    class_names,
    num_classes=len(class_names),
    epochs=epochs,
    batch_size=batch_size,
    num_folds=num_folds
)