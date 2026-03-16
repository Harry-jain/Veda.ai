"""
evaluate.py
===========
Evaluates a saved model on the test set and optionally classifies a single image.

Usage
-----
    # Full test-set evaluation
    python evaluate.py --model models/plant_cnn_final.h5 --test prepared_data/Test

    # Single-image prediction
    python evaluate.py --model models/plant_cnn_final.h5 --image path/to/leaf.jpg

    # Both at once
    python evaluate.py --model models/plant_cnn_final.h5 \
                       --test prepared_data/Test \
                       --image path/to/leaf.jpg \
                       --cm                        # also plot confusion matrix

Dependencies
------------
    tensorflow>=2.10  numpy  matplotlib  scikit-learn  tqdm  Pillow
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = (256, 256)
BATCH_SIZE = 11


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_keras_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model from {model_path} …")
    return load_model(str(model_path))


def evaluate_on_test_set(model: tf.keras.Model, test_dir: Path,
                         plot_cm: bool = False) -> None:
    """Runs evaluation + optional confusion matrix."""
    gen = ImageDataGenerator(rescale=1.0 / 255.0)
    flow = gen.flow_from_directory(
        str(test_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    print(f"\nTest images  : {flow.samples}")
    print(f"Test classes : {flow.num_classes}")

    loss, acc = model.evaluate(flow, steps=len(flow), verbose=1)
    print(f"\n{'─'*40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'─'*40}\n")

    if plot_cm:
        try:
            from sklearn.metrics import confusion_matrix, classification_report
        except ImportError:
            print("[WARN] scikit-learn not installed — skipping confusion matrix.")
            return

        print("Generating predictions for confusion matrix …")
        y_true = flow.classes
        y_pred = model.predict(flow, steps=len(flow), verbose=1).argmax(axis=1)

        labels = list(flow.class_indices.keys())
        cm     = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(tick_marks); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label',      fontsize=12)
        ax.set_title('Confusion Matrix — Medicinal Plant CNN', fontweight='bold', fontsize=13)

        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=5,
                        color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        cm_path = Path('graphs/confusion_matrix.png')
        cm_path.parent.mkdir(exist_ok=True)
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved → {cm_path}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))


def predict_single(model: tf.keras.Model, image_path: Path,
                   class_names: list[str] | None = None) -> None:
    """Run inference on one image and print top-5 predictions."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = img.convert('RGB').resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)           # (1, 256, 256, 3)

    preds  = model.predict(arr, verbose=0)[0]       # (num_classes,)
    top5   = preds.argsort()[::-1][:5]

    print(f"\nPredictions for: {image_path.name}")
    print(f"{'─'*45}")
    for rank, idx in enumerate(top5, 1):
        name = class_names[idx] if class_names else f"class_{idx}"
        print(f"  {rank}. {name:<25s}  {preds[idx]*100:6.2f}%")
    print(f"{'─'*45}")

    best = top5[0]
    name = class_names[best] if class_names else f"class_{best}"
    print(f"\n✓ Predicted plant: {name}  (confidence {preds[best]*100:.2f}%)\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate / infer with medicinal plant CNN.')
    parser.add_argument('--model', required=True,
                        help='Path to saved .h5 model.')
    parser.add_argument('--test', default=None,
                        help='Path to test directory for full evaluation.')
    parser.add_argument('--image', default=None,
                        help='Path to a single leaf image for inference.')
    parser.add_argument('--cm', action='store_true',
                        help='Plot and save a confusion matrix (requires scikit-learn).')
    parser.add_argument('--classes', default=None,
                        help='Optional comma-separated list of class names (for single-image mode).')
    args = parser.parse_args()

    model = load_keras_model(Path(args.model))

    # Parse optional class list
    class_names: list[str] | None = None
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(',')]

    if args.test:
        evaluate_on_test_set(model, Path(args.test), plot_cm=args.cm)

    if args.image:
        predict_single(model, Path(args.image), class_names=class_names)

    if not args.test and not args.image:
        parser.error("Provide at least one of --test or --image.")


if __name__ == '__main__':
    main()
