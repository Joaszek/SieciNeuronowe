import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from utils import make_dataset, categorical_focal_loss, make_alpha, MacroF1Checkpoint
from model import create_model
from collections import Counter
from glob import glob
from collections import defaultdict
import math
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


TRAIN_CHANNEL = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

def find_backbone(m):
    # 1) spróbuj po nazwie (tak zwykle nazywa się EfficientNetB0)
    for name in ("efficientnetb0", "EfficientNetB0"):
        try:
            return m.get_layer(name)
        except Exception:
            pass
    # 2) warstwa, która sama jest modelem i ma podwarstwy
    for l in m.layers:
        if isinstance(l, tf.keras.Model) and hasattr(l, "layers") and len(l.layers) > 0:
            # EfficientNetB0 zwykle ma dużo warstw i nazwę zaczynającą się na efficientnet
            if l.name.lower().startswith("efficientnet"):
                return l
    # 3) ostatnia deska ratunku: pierwsza warstwa, która ma .layers
    for l in m.layers:
        if hasattr(l, "layers") and len(l.layers) > 0:
            return l
    raise ValueError("Nie mogę znaleźć backbone'u (EfficientNetB0) w modelu.")

def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    # odwrotność częstości (łagodnie)
    weights = {i: total / (num_classes * max(1, c)) for i, c in enumerate(counts)}
    return weights

def upsample_to_balance(files, labels_np, class_names, target_multiple=1.0):
    # policz licznosci
    counts = defaultdict(int)
    for lab in labels_np:
        counts[int(lab)] += 1
    max_count = max(counts.values())
    # można lekko "niedobalansować" docelowo (np. 0.7 * max_count), by nie przesadzić
    target = int(max_count * target_multiple)

    # grupuj pliki per klasa
    by_class = defaultdict(list)
    for f, lab in zip(files, labels_np):
        by_class[int(lab)].append(f)

    # zbuduj nową, zbalansowaną listę
    balanced_files, balanced_labels = [], []
    for cls_idx, lst in by_class.items():
        if len(lst) == 0:
            continue
        # powielaj z losowaniem do target
        mul = math.ceil(target / len(lst))
        expanded = (lst * mul)[:target]
        random.shuffle(expanded)
        balanced_files.extend(expanded)
        balanced_labels.extend([cls_idx] * len(expanded))

    # przetasuj globalnie
    mix = list(zip(balanced_files, balanced_labels))
    random.shuffle(mix)
    balanced_files, balanced_labels = zip(*mix)
    return list(balanced_files), np.array(balanced_labels, dtype=np.int32)


def train_kfold(all_files, class_names, num_classes=4, epochs=10, batch_size=32, num_folds=5):
    if len(all_files) == 0:
        raise ValueError("Brak obrazów po filtracji. Sprawdź strukturę katalogów i rozszerzenia plików.")

    best_acc = 0.0
    best_model_path = '/opt/ml/model/best_model'
    fold_results = []

    labels = np.array([class_names.index(os.path.basename(os.path.dirname(f))) for f in all_files])
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(kf.split(all_files, labels)):
        print(f'\n>>> Fold {i + 1}/{num_folds}')
        train_files = [all_files[idx] for idx in train_index]
        val_files   = [all_files[idx] for idx in val_index]

        # rozkład klas
        def dist(idxs):
            u, c = np.unique(labels[idxs], return_counts=True)
            return dict(zip([class_names[u_] for u_ in u], c))
        print(f"Fold {i + 1}: train={len(train_files)}, val={len(val_files)}")
        print("Train class distribution:", dist(train_index))
        print("Val class distribution:", dist(val_index))


        train_files_bal, train_labels_bal = upsample_to_balance(
            train_files, labels[train_index], class_names, target_multiple=0.5  # 80% max_count
        )

        train_ds = make_dataset(
            train_files_bal, class_names, labels=train_labels_bal,
            batch_size=batch_size, shuffle=True
        )
        val_ds   = make_dataset(
            val_files, class_names, labels=labels[val_index],
            batch_size=batch_size, shuffle=False
        )

        # --- wagi klas liczymy na oryginalnych etykietach train (przed upsamplingiem) ---
        train_labels_np = labels[train_index]
        class_weights = compute_class_weights(train_labels_np, num_classes)

        # --- model + referencja do bazy (EfficientNetB0 jest 1. warstwą w Sequential) ---
        model = create_model(num_classes=num_classes)
        base_model = find_backbone(model)

        # --- callbacki ---
        ckpt_dir = f'/opt/ml/model/fold_{i+1}'
        os.makedirs(ckpt_dir, exist_ok=True)

        callbacks = [
            MacroF1Checkpoint(val_ds, os.path.join(ckpt_dir, "best_f1.h5")),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1, cooldown=1),
            EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ]


# ===== FAZA 1: WARM-UP (zamrożona baza, opcjonalnie bez mixup i bez class_weight) =====

        alpha_vec = make_alpha(labels[train_index], num_classes)
        loss_fn = categorical_focal_loss(gamma=2.0, alpha=alpha_vec, label_smoothing=0.05)
        base_model.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=loss_fn,
            metrics=['accuracy']
        )

        warmup_epochs = max(2, epochs // 3)
        finetune_epochs = max(0, epochs - warmup_epochs)

        # Jeśli chcesz bardziej stabilnego startu, możesz na czas warm-up wyłączyć class_weight:
        use_class_weight_warmup = False
        cw_warmup = class_weights if use_class_weight_warmup else None

        _ = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            verbose=1,
            callbacks=callbacks,
            # class_weight=cw_warmup
        )

        os.environ["USE_MIXUP"] = "1"

        # Zbuduj nowe train_ds z mixupem na fazę fine-tune
        train_ds = make_dataset(
            train_files_bal, class_names, labels=train_labels_bal,
            batch_size=batch_size, shuffle=True
        )

        # ===== FAZA 2: FINE-TUNE (bezpiecznie) =====
        base_model.trainable = True

        # BatchNorm ZAMROŻONE (ważne na małej walidacji)
        for l in base_model.layers:
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False

        num_layers = len(base_model.layers)
        unfreeze_last = max(1, int(num_layers * 0.20))   # 20% top (było 35%)
        for layer in base_model.layers[:-unfreeze_last]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),  # mniejszy LR + clipping
            loss=loss_fn,
            metrics=['accuracy']
        )

        _ = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            verbose=1,
            callbacks=callbacks
        )


        # ===== EWALUACJA =====
        val_preds = model.predict(val_ds, verbose=0)
        val_true_classes = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds], axis=0)
        val_preds_classes = np.argmax(val_preds, axis=1)

        acc = np.mean(val_preds_classes == val_true_classes)
        print(f'Fold {i + 1} Accuracy: {acc:.4f}')

        cm = confusion_matrix(val_true_classes, val_preds_classes)
        print(f'Fold {i + 1} Confusion Matrix:\n{cm}')

        report = classification_report(val_true_classes, val_preds_classes, target_names=class_names)
        print(f'Classification Report for Fold {i + 1}:\n{report}')

        prec = precision_score(val_true_classes, val_preds_classes, average="macro", zero_division=0)
        rec = recall_score(val_true_classes, val_preds_classes, average="macro", zero_division=0)
        f1 = f1_score(val_true_classes, val_preds_classes, average="macro", zero_division=0)
        print(f'Fold {i + 1} Summary: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}')
        fold_results.append((acc, prec, rec, f1))

        per_class_rec = recall_score(val_true_classes, val_preds_classes, average=None, zero_division=0)
        for c, r in zip(class_names, per_class_rec):
            print(f"Recall[{c}]: {r:.3f}")

        if acc > best_acc:
            best_acc = acc
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            model.save(best_model_path)
            print(f'>>> New best model: {best_model_path} (accuracy={best_acc:.4f})')



    print(f'\nBest final model saved in: {best_model_path} with accuracy={best_acc:.4f}')
    print("\n===== TRAINING SUMMARY =====")
    for i, (acc, prec, rec, f1) in enumerate(fold_results, 1):
        print(f"Fold {i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    avg_acc = np.mean([r[0] for r in fold_results])
    avg_prec = np.mean([r[1] for r in fold_results])
    avg_rec = np.mean([r[2] for r in fold_results])
    avg_f1 = np.mean([r[3] for r in fold_results])
    print("\nAVG RESULTS:")
    print(f"Accuracy:  {avg_acc:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1-score:  {avg_f1:.4f}")

# ======= GŁÓWNY BLOK =======

# Zbierz wszystkie obrazy rekurencyjnie (różne wielkości liter w rozszerzeniach)
exts = ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG")
all_files_raw = []
for e in exts:
    all_files_raw.extend(glob(os.path.join(TRAIN_CHANNEL, "**", e), recursive=True))

if len(all_files_raw) == 0:
    # pokaż pomocną diagnostykę
    print(">>> DEBUG listing of TRAIN channel root:", TRAIN_CHANNEL)
    for root, dirs, files in os.walk(TRAIN_CHANNEL):
        print(root, "->", files[:5])
    raise ValueError(f"Nie znaleziono żadnych obrazów w {TRAIN_CHANNEL}. Upewnij się, że struktura to train/<klasa>/*.jpg")

labels_all = [os.path.basename(os.path.dirname(f)) for f in all_files_raw]
counts_all = Counter(labels_all)

# wybieramy top-4 najliczniejsze klasy (jeżeli jest mniej niż 4, weź tyle ile jest)
topk = min(4, len(counts_all))
top_classes = [cls for cls, _ in counts_all.most_common(topk)]

# przefiltruj pliki tylko do tych klas
all_files = [f for f in all_files_raw if os.path.basename(os.path.dirname(f)) in top_classes]
class_names = sorted(top_classes)
print("Final class names:", class_names)

# policz rozkład po filtracji (dla doboru liczby foldów)
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
