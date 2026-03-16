"""
train.py
========
Trains a CNN for medicinal-plant identification.

Architecture
------------
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
  → Flatten → Dense(128, relu) → Dropout(0.5) → Dense(N_CLASSES, softmax)

Input  : 256 × 256 RGB images, values rescaled to [0, 1]
Output : saved model at  models/plant_cnn.h5
         training curves at  graphs/training_accuracy.png
                              graphs/training_loss.png

Usage
-----
    python train.py --train prepared_data/Train --val prepared_data/Val --epochs 50

Dependencies
------------
    tensorflow>=2.10  matplotlib  numpy  tqdm
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 11
EPOCHS      = 50
LR          = 1e-3


def build_model(num_classes: int) -> tf.keras.Model:
    """Exact architecture from the original research notebook."""
    model = models.Sequential([
        layers.Conv2D(32,  (3, 3), activation='relu', padding='same',
                      input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64,  (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def make_generators(train_dir: Path, val_dir: Path):
    """Create augmented train and plain val data generators."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_flow = train_gen.flow_from_directory(
        str(train_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED,
    )
    val_flow = val_gen.flow_from_directory(
        str(val_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    return train_flow, val_flow


def plot_history(history: tf.keras.callbacks.History, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history['accuracy'],     label='Training Accuracy',   color='#2196F3', lw=1.8)
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF5722', lw=1.8,
            linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy — Medicinal Plant CNN', fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_accuracy.png', dpi=150)
    plt.close()

    # Loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history['loss'],     label='Training Loss',   color='#4CAF50', lw=1.8)
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#9C27B0', lw=1.8,
            linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Cat. Cross-Entropy)')
    ax.set_title('Model Loss — Medicinal Plant CNN', fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_loss.png', dpi=150)
    plt.close()

    print(f"  Plots saved to {out_dir}/")


def train(train_dir: Path, val_dir: Path, model_out: Path, graph_out: Path) -> None:
    train_flow, val_flow = make_generators(train_dir, val_dir)
    num_classes = train_flow.num_classes
    print(f"\nClasses detected : {num_classes}")
    print(f"Training images  : {train_flow.samples}")
    print(f"Validation images: {val_flow.samples}\n")

    model = build_model(num_classes)
    model.summary()

    model_out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            str(model_out / 'best_model.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_flow,
        steps_per_epoch=len(train_flow),
        epochs=EPOCHS,
        validation_data=val_flow,
        validation_steps=len(val_flow),
        callbacks=callbacks,
    )

    # Save final model
    final_path = model_out / 'plant_cnn_final.h5'
    model.save(str(final_path))
    print(f"\nFinal model saved → {final_path}")

    plot_history(history, graph_out)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='Train medicinal plant CNN.')
    parser.add_argument('--train',  default='prepared_data/Train',
                        help='Path to training directory (default: prepared_data/Train).')
    parser.add_argument('--val',    default='prepared_data/Val',
                        help='Path to validation directory (default: prepared_data/Val).')
    parser.add_argument('--models', default='models',
                        help='Output directory for saved models (default: models).')
    parser.add_argument('--graphs', default='graphs',
                        help='Output directory for training plots (default: graphs).')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS}).')
    parser.add_argument('--batch',  type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE}).')
    parser.add_argument('--lr',     type=float, default=LR,
                        help=f'Initial Adam learning rate (default: {LR}).')
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE, LR
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    LR         = args.lr

    train(
        train_dir  = Path(args.train),
        val_dir    = Path(args.val),
        model_out  = Path(args.models),
        graph_out  = Path(args.graphs),
    )


if __name__ == '__main__':
    main()
