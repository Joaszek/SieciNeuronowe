from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def class_weight():
    class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    class_counts = [6705, 1113, 1099, 514, 327, 142, 115]
    class_indices = dict(enumerate(class_names))

    class_weight_values = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_names)),
        y=np.concatenate([
            np.full(count, i) for i, count in enumerate(class_counts)
        ])
    )

    return dict(enumerate(class_weight_values))
