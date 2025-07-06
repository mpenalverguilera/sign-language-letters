#!/usr/bin/env python
"""
filter_dataset.py  ─ Clean / reject ASL-alphabet images with MediaPipe + multiprocessing.

Output structure
────────────────
<output>/clean/<class>/....jpg     ← images that meet the rules
<output>/rejected/<class>/....jpg  ← images that fail the rules

Rules
─────
• Letter classes (A-Z, space, delete)  → keep  if ≥ 1 hand detected.
• "nothing" class                     → keep  if NO hands detected.

CLI examples
────────────
python filter_dataset.py                       # defaults + progress bar
python filter_dataset.py --no-progress         # silent
python filter_dataset.py --input data/raw --output data/filtered
"""

# ── Std-lib & third-party imports ──────────────────────────────────────────────
import os
import sys
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import mediapipe as mp
from tqdm import tqdm
import random

# ───────────────────────────────────────────────────────────────────────────────
# Globals – one MediaPipe Hands instance per *process*
# ───────────────────────────────────────────────────────────────────────────────
MP_HANDS = None  # will be created inside each worker via _init_worker()


def _init_worker() -> None:
    """Create exactly one MediaPipe Hands object inside each worker process."""
    global MP_HANDS
    MP_HANDS = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────
def iterate_images(root: Path) -> Iterable[Path]:
    """Yield every *.jpg file under *root* (recursive, memory-efficient)."""
    yield from root.rglob("*.jpg")


def detect_hand(img_path: Path) -> bool:
    """Return True if ≥ 1 hand is detected in *img_path* (uses global MP_HANDS)."""
    global MP_HANDS
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = MP_HANDS.process(img_rgb)
    return bool(result.multi_hand_landmarks)


def decide_validity(img_path: Path) -> Tuple[Path, bool]:
    """
    Decide if *img_path* should go to 'clean' or 'rejected' according to rules.
    Returns (Path, is_valid).
    """
    cls = img_path.parent.name.lower()
    hand = detect_hand(img_path)

    is_valid = (not hand) if cls == "nothing" else hand
    return img_path, is_valid


def copy_image(src: Path, dst: Path) -> None:
    """Copy *src* → *dst*, creating parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ───────────────────────────────────────────────────────────────────────────────
# Main routine
# ───────────────────────────────────────────────────────────────────────────────
def filter_dataset(input_dir: Path, output_dir: Path, show_progress: bool) -> None:
    # Prepare output folders
    clean_root = output_dir / "clean"
    rej_root = output_dir / "rejected"
    clean_root.mkdir(parents=True, exist_ok=True)
    rej_root.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    img_iterable = list(iterate_images(input_dir)) if show_progress else iterate_images(
        input_dir
    )
    n_total = len(img_iterable) if show_progress else None
    if n_total is not None and n_total == 0:
        raise RuntimeError("No *.jpg images found in input directory.")

    # Workers
    n_workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"[INFO] Using {n_workers} worker processes")

    # chunk-size ≈ (total / workers) ─ less overhead than one task per image
    chunk = 500

    # Progress bar
    pbar = tqdm(total=n_total, desc="Filtering images", unit="img", file=sys.stdout) if show_progress else None

    kept = rejected = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for img_path, is_valid in ex.map(decide_validity, img_iterable, chunksize=chunk):
            dst_dir = clean_root if is_valid else rej_root
            copy_image(img_path, dst_dir / img_path.parent.name / img_path.name)
            kept += is_valid
            rejected += not is_valid
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # Summary
    total = kept + rejected
    rej_pct = (rejected / total * 100) if total else 0.0
    print("\n──────────── Summary ────────────")
    print(f"Images kept     : {kept}")
    print(f"Images rejected : {rejected}")
    print(f"Total processed : {total}")
    print(f"Reject rate     : {rej_pct:.2f}%")
    print(f"Workers used    : {n_workers}")
    print("────────────────────────────────")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter the ASL alphabet dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/raw/asl_alphabet_train"),
        help="Raw dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/filtered"),
        help="Destination directory",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    args = parser.parse_args()

    print("[INFO] Filtering dataset …")
    filter_dataset(args.input, args.output, show_progress=not args.no_progress)
    print("[INFO] Finished.")
