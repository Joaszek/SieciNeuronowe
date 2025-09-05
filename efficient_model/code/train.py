import os, json, time, math, random
import tensorflow as tf
from model import create_model
from utils import load_data, scan_dir_return_paths_labels, make_tfds_from_paths
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, logging

os.environ["PYTHONUNBUFFERED"] = "1"
os.makedirs("/opt/ml/output/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("/opt/ml/output/logs/training.log")]
)
log = logging.getLogger("train")
log.info("Booting training script...")



def compute_class_weights_from_labels(y, n_classes):
    counts = defaultdict(int)
    for yi in y:
        counts[int(yi)] += 1
    total = len(y)
    weights = {}
    for i in range(n_classes):
        ci = counts.get(i, 0)
        weights[i] = total / (n_classes * max(ci, 1))
    print("Class counts:", dict(counts))
    print("Class weights:", weights)

    log.info(f"Class counts: {dict(counts)}")
    log.info(f"Class weights: {weights}")
    return weights


def evaluate_and_save_results(model, val_ds, class_names, fold, output_path):
    # 1) Predykcje i etykiety
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # 2) Raport (słownik do wyjęcia metryk)
    rep_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    rep_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=False
    )

    # 3) Macierz pomyłek (list -> JSON-friendly)
    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    # 4) Wybór metryk do raportu folda (tu: macro avg – możesz zmienić na weighted avg)
    acc  = float(rep_dict.get("accuracy", 0.0))
    prec = float(rep_dict["macro avg"]["precision"])
    rec  = float(rep_dict["macro avg"]["recall"])
    f1   = float(rep_dict["macro avg"]["f1-score"])

    # 5) Czytelny wydruk jak na zrzutach
    print(f"\nFold {fold} Confusion Matrix:\n{cm_list}")
    print(f"Fold {fold} Summary: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}\n")

    log.info(f"\nFold {fold} Confusion Matrix:\n{cm_list}")
    log.info(f"Fold {fold} Summary: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}\n")

    # 6) Zapis tekstowego raportu i obrazka
    with open(os.path.join(output_path, f"fold{fold}_report.txt"), "w") as f:
        f.write(rep_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_path, f"fold{fold}_confusion_matrix.png"))
    plt.close()

    # 7) Zapis JSON z metrykami + CM
    fold_json = {
        "fold": fold,
        "accuracy_macro": acc,          # acc ogólne (Keras/CLF report)
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm_list,
        "class_names": class_names
    }
    with open(os.path.join(output_path, f"fold{fold}_results.json"), "w") as f:
        json.dump(fold_json, f, indent=2)

    
    print(f"\n===== Fold {fold} Classification Report =====\n{rep_text}", flush=True)
    print(f"Fold {fold} Confusion Matrix:\n{cm_list}", flush=True)
    print(f"Fold {fold} Summary: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}\n", flush=True)
    print("Fold json: ", fold_json)

    log.info(f"\n===== Fold {fold} Classification Report =====\n{rep_text}")
    log.info(f"Fold {fold} Confusion Matrix:\n{cm_list}")
    log.info(f"Fold {fold} Summary: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    log.info(f"===== Fold {fold} Classification Report =====\n{rep_text}")
    log.info(f"Fold json: {fold_json}")

    return fold_json


train_path = '/opt/ml/input/data/train'
val_path   = '/opt/ml/input/data/val'
output_path = '/opt/ml/model'

while not os.path.exists(train_path):
    time.sleep(1)

epochs     = int(os.environ.get('EPOCHS', 10))
batch_size = int(os.environ.get('BATCH_SIZE', 32))
k_folds    = int(os.environ.get('K_FOLDS', 1))
seed       = 42
random.seed(seed)

# ------- Build the full pool of samples (train + val) -------
paths_tr, y_tr, classes_tr = scan_dir_return_paths_labels(train_path)
paths_va, y_va, classes_va = scan_dir_return_paths_labels(val_path)

# Sanity: prefer non-empty class mapping; assume both dirs share the same set
class_names = classes_tr if classes_tr else classes_va
all_paths = paths_tr + paths_va
all_labels = y_tr + y_va

if k_folds <= 1 or len(all_paths) == 0:
    # fallback to your original single-split training
    print("K_FOLDS<=1 or no data detected: running standard train/val.")
    log.info("K_FOLDS<=1 or no data detected: running standard train/val.")
    from collections import Counter
    def compute_class_weights(root):
        class_names_local = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        counts = {}
        for c in class_names_local:
            counts[c] = len(os.listdir(os.path.join(root, c)))
        total = sum(counts.values())
        n_classes_local = len(class_names_local)
        weights = {i: total / (n_classes_local * counts[c]) for i, c in enumerate(class_names_local)}
        print("Class counts:", counts)
        print("Class weights:", weights)

        log.info(f"Class counts: {dict(counts)}")
        log.info(f"Class weights: {weights}")
        return weights

    for name in os.listdir(train_path):
        class_path = os.path.join(train_path, name)
        if os.path.isdir(class_path):
            print(f"{name} → {os.listdir(class_path)[:3]}")
            log.info(f"{name} → {os.listdir(class_path)[:3]}")

    train_ds = load_data(train_path, batch_size=batch_size, training=True)
    val_ds   = load_data(val_path,   batch_size=batch_size, training=False)
    class_weight = compute_class_weights(train_path)

    model = create_model(num_classes=7)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_path, 'best.weights.h5'),
            save_best_only=True, save_weights_only=True,
            monitor='val_accuracy', mode='max'
        ),
    ]
    recalls = [tf.keras.metrics.Recall(class_id=i, name=f'recall_c{i}') for i in range(7)]
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'] + recalls)
    model.fit(train_ds, validation_data=val_ds,
              epochs=max(epochs // 2, 3), class_weight=class_weight, callbacks=callbacks)

    base = model.layers[0]
    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < int(0.8 * len(base.layers)):
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss=loss, metrics=['accuracy'] + recalls)
    model.fit(train_ds, validation_data=val_ds,
              epochs=max(epochs - max(epochs // 2, 3), 3),
              class_weight=class_weight, callbacks=callbacks)

    model.save(output_path)
    raise SystemExit(0)

# ----------------- Stratified K-Fold -----------------
print(f"Running Stratified {k_folds}-Fold CV on {len(all_paths)} images.")
log.info(f"Running Stratified {k_folds}-Fold CV on {len(all_paths)} images.")

num_classes = len(set(all_labels))
idxs_by_class = defaultdict(list)
for idx, yi in enumerate(all_labels):
    idxs_by_class[int(yi)].append(idx)

# shuffle per-class
for c in idxs_by_class:
    random.shuffle(idxs_by_class[c])

# build folds per class, then merge
fold_bins = [list() for _ in range(k_folds)]
for c, idxs in idxs_by_class.items():
    # split roughly equal chunks
    chunk_size = max(1, len(idxs) // k_folds)
    for f in range(k_folds):
        start = f * chunk_size
        end   = (f+1) * chunk_size if f < k_folds - 1 else len(idxs)
        fold_bins[f].extend(idxs[start:end])

histories_summary = []

for fold in range(k_folds):
    val_idx = set(fold_bins[fold])
    train_idx = [i for i in range(len(all_paths)) if i not in val_idx]

    tr_paths = [all_paths[i] for i in train_idx]
    tr_labels= [all_labels[i] for i in train_idx]
    va_paths = [all_paths[i] for i in val_idx]
    va_labels= [all_labels[i] for i in val_idx]

    print(f"Fold {fold+1}/{k_folds}: train={len(tr_paths)}, val={len(va_paths)}")
    log.info(f"Fold {fold+1}/{k_folds}: train={len(tr_paths)}, val={len(va_paths)}")


    train_ds = make_tfds_from_paths(tr_paths, tr_labels, batch_size=batch_size, training=True)
    val_ds   = make_tfds_from_paths(va_paths, va_labels, batch_size=batch_size, training=False)

    class_weight = compute_class_weights_from_labels(tr_labels, n_classes=num_classes)

    model = create_model(num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_path, f'fold{fold+1}.weights.h5'),
            save_best_only=True, save_weights_only=True,
            monitor='val_accuracy', mode='max'
        ),
    ]
    recalls = [tf.keras.metrics.Recall(class_id=i, name=f'recall_c{i}') for i in range(num_classes)]
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # Phase 1: head training
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=['accuracy'] + recalls)
    hist1 = model.fit(train_ds, validation_data=val_ds,
                      epochs=max(epochs // 2, 3), class_weight=class_weight, callbacks=callbacks)

    # Phase 2: fine-tune tail of the base model
    base = model.layers[0]
    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < int(0.8 * len(base.layers)):
            layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss=loss, metrics=['accuracy'] + recalls)
    hist2 = model.fit(train_ds, validation_data=val_ds,
                      epochs=max(epochs - max(epochs // 2, 3), 3),
                      class_weight=class_weight, callbacks=callbacks)

    # Save the whole model per fold (optional, heavier)
    model.save(os.path.join(output_path, f'fold{fold+1}'))

    fold_stats = evaluate_and_save_results(model, val_ds, class_names, fold+1, output_path)

    # Dołóż do metryk także najlepsze val_acc z treningu
    final_metrics = {
        "fold": fold + 1,
        "best_val_accuracy": max(hist1.history.get('val_accuracy', [0]) +
                                hist2.history.get('val_accuracy', [0])),
        "epochs_run": len(hist1.history.get('accuracy', [])) +
                    len(hist2.history.get('accuracy', [])),
        # scalone metryki z ewaluacji:
        "accuracy_macro": fold_stats["accuracy_macro"],
        "precision_macro": fold_stats["precision_macro"],
        "recall_macro": fold_stats["recall_macro"],
        "f1_macro": fold_stats["f1_macro"]
    }
    histories_summary.append(final_metrics)
    with open(os.path.join(output_path, f'fold{fold+1}_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)


# overall summary (średnie po foldach)
if histories_summary:
    avg = lambda k: float(np.mean([m[k] for m in histories_summary]))
    overall = {
        "k_folds": k_folds,
        "classes": class_names,
        "avg_best_val_accuracy": avg("best_val_accuracy"),
        "avg_accuracy_macro":     avg("accuracy_macro"),
        "avg_precision_macro":    avg("precision_macro"),
        "avg_recall_macro":       avg("recall_macro"),
        "avg_f1_macro":           avg("f1_macro"),
        "folds": histories_summary
    }
else:
    overall = {"k_folds": k_folds, "classes": class_names, "folds": histories_summary}

with open(os.path.join(output_path, 'kfold_summary.json'), 'w') as f:
    json.dump(overall, f, indent=2)
