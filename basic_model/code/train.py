from model import create_model
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import os
import shutil
from utils import load_data
from class_weights import class_weight


def train_kfold(all_files, class_names, num_classes=7, epochs=10, batch_size=32, num_folds=5):
    """
    Trains a classification model using stratified K-fold cross-validation.

    Parameters
    ----------
    all_files : list of str
        List of file paths to all training images.
    class_names : list of str
        List of class labels corresponding to the dataset.
    num_classes : int, default=7
        Number of unique classes in the dataset.
    epochs : int, default=10
        Number of epochs to train each fold.
    batch_size : int, default=32
        Batch size for training and validation datasets.
    num_folds : int, default=5
        Number of folds for stratified cross-validation.

    Returns
    -------
    None
        Saves the best-performing model to disk and prints performance
        metrics (accuracy, precision, recall, F1-score) for each fold
        as well as their averages.
    """
    best_acc = 0.0
    best_model_path = '/opt/ml/model/best_model'
    fold_results = []

    labels = np.array([class_names.index(os.path.basename(os.path.dirname(f))) for f in all_files])

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    #labels = [class_names.index(os.path.basename(os.path.dirname(f))) for f in all_files]

    for i, (train_index, val_index) in enumerate(kf.split(all_files, labels)):
        print(f'\n>>> Fold {i + 1}/{num_folds}')
        train_files = [all_files[idx] for idx in train_index]
        val_files = [all_files[idx] for idx in val_index]

        print(f"Fold {i + 1}: train={len(train_files)}, val={len(val_files)}")
        unique, counts = np.unique(labels[train_index], return_counts=True)
        print("Train class distribution:", dict(zip([class_names[u] for u in unique], counts)))
        unique, counts = np.unique(labels[val_index], return_counts=True)
        print("Val class distribution:", dict(zip([class_names[u] for u in unique], counts)))

        train_ds = load_data(train_files, class_names, batch_size=batch_size)
        val_ds = load_data(val_files, class_names, batch_size=batch_size)

        model = create_model(num_classes=num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        cw = class_weight()
        fold_class_weights = {i: cw[i] for i in range(num_classes)}

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=fold_class_weights,
            verbose=1
        )

        val_preds = model.predict(val_ds, verbose=0)
        val_true_classes = np.concatenate([np.argmax(y, axis=1) for x, y in val_ds], axis=0)
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

train_dir = '/opt/ml/input/data/training'
all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(train_dir)
             for f in filenames if f.endswith(('.jpg', '.png', '.jpeg'))]

class_names = sorted(next(os.walk(train_dir))[1])
print(class_names)

epochs = int(os.environ.get('EPOCHS', 10))
batch_size = int(os.environ.get('BATCH_SIZE', 32))

train_kfold(all_files, class_names, num_classes=len(class_names), epochs=epochs, batch_size=batch_size)
