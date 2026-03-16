"""
data_prep.py
============
Prepares the raw Kaggle dataset for training.

Steps
-----
1. Reads images from KAGGLE_SRC (one sub-folder per class).
2. Resizes every image to TARGET_SIZE.
3. Splits into train / val / test at an 80 / 10 / 10 ratio.
4. Writes the split images to OUTPUT_ROOT/{Train,Val,Test}/<class>/.

Usage
-----
    python data_prep.py --src /path/to/kaggle/dataset --out prepared_data

Kaggle dataset used in the original project
--------------------------------------------
"Indian Medicinal Leaves Image Dataset" — 26 plant classes, ~15k images.
https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-image-datasets
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


TARGET_SIZE   = (256, 256)
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (remaining 10 %)
VALID_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
RANDOM_SEED   = 42


def resize_and_copy(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Open, resize (Lanczos), and save an image."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        img.save(dst)


def split_and_prepare(src_root: Path, out_root: Path) -> None:
    random.seed(RANDOM_SEED)

    splits = ('Train', 'Val', 'Test')
    for s in splits:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in sorted(src_root.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No sub-directories found in {src_root}. "
            "Make sure the dataset is structured as <src>/<class_name>/<image>."
        )

    total_copied = 0

    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir()
                 if f.suffix.lower() in VALID_EXTS]
        if not files:
            print(f"  [WARN] No images in {class_dir.name} — skipping.")
            continue

        random.shuffle(files)
        n          = len(files)
        n_train    = int(TRAIN_RATIO * n)
        n_val      = int(VAL_RATIO   * n)

        buckets = {
            'Train': files[:n_train],
            'Val':   files[n_train : n_train + n_val],
            'Test':  files[n_train + n_val :],
        }

        for split_name, split_files in buckets.items():
            desc = f"{class_dir.name}/{split_name}"
            for f in tqdm(split_files, desc=desc, unit='img', leave=False):
                dst = out_root / split_name / class_dir.name / f.name
                resize_and_copy(f, dst, TARGET_SIZE)
                total_copied += 1

        print(f"  {class_dir.name:30s}  "
              f"train={len(buckets['Train'])}  "
              f"val={len(buckets['Val'])}  "
              f"test={len(buckets['Test'])}")

    print(f"\nDone. {total_copied} images written to {out_root}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare medicinal-plant dataset.")
    parser.add_argument('--src', required=True,
                        help='Path to raw Kaggle dataset root (one folder per class).')
    parser.add_argument('--out', default='prepared_data',
                        help='Output root directory (default: prepared_data).')
    parser.add_argument('--size', type=int, default=256,
                        help='Target image size in pixels (square). Default: 256.')
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    size     = (args.size, args.size)

    if not src_root.exists():
        raise FileNotFoundError(f"Source path not found: {src_root}")

    print(f"Source   : {src_root}")
    print(f"Output   : {out_root}")
    print(f"Size     : {size}")
    print(f"Split    : {int(TRAIN_RATIO*100)} / {int(VAL_RATIO*100)} / "
          f"{int((1-TRAIN_RATIO-VAL_RATIO)*100)}  (train/val/test)")
    print()

    global TARGET_SIZE
    TARGET_SIZE = size

    split_and_prepare(src_root, out_root)


if __name__ == '__main__':
    main()
