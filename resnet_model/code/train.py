import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from utils import make_dataset
from model import create_resnet_model
from collections import Counter
from glob import glob

# UŻYWAJ zmiennej z SageMaker zamiast na sztywno
TRAIN_CHANNEL = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

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

        train_ds = make_dataset(train_files, class_names, labels=labels[train_index], batch_size=batch_size, shuffle=True)
        val_ds   = make_dataset(val_files, class_names, labels=labels[val_index], batch_size=batch_size, shuffle=False)

        model = create_resnet_model(num_classes=num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

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
