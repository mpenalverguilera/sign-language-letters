#!/usr/bin/env python
"""
vectorise_dataset.py
────────────────────
Convert a filtered ASL‐alphabet image dataset into flat normalised landmark
vectors (60-D) and store them in CSV files ready for ML/NN training.

Outputs:
    features_all.csv  # every sample
    train.csv         # 80 % stratified
    val.csv           # 10 % stratified
    test.csv          # 10 % stratified
"""

from __future__ import annotations
import os, sys, csv, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import StratifiedShuffleSplit

# ───────────────────────── helper: iterate images ────────────────────────────
def iterate_images(root: Path):
    """Yield every *.jpg file under *root* recursively."""
    yield from root.rglob("*.jpg")

# ───────────────────────── worker initialisation ─────────────────────────────
MP_HANDS = None

def _init_worker() -> None:
    global MP_HANDS
    MP_HANDS = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

# ───────────────────────── single-image → vector ─────────────────────────────
def one_image_to_vector(img_path: Path) -> Tuple[str, List[float]] | Tuple[str, str]:
    """
    Return (label, 60-D vector) or error message if:
      • class == 'nothing'
      • no hand detected
      • image corrupted
    """
    label = img_path.parent.name
    if label == "nothing":
        return label, "class_nothing"

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return label, "corrupt_image"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lm_res = MP_HANDS.process(img_rgb)
    if not lm_res.multi_hand_landmarks:
        return label, "no_hand"

    lm = np.array([[p.x, p.y, p.z] for p in lm_res.multi_hand_landmarks[0].landmark],
                  dtype=np.float32)
    lm -= lm[0]                                   # centre at wrist
    scale = np.max(np.ptp(lm, axis=0)) or 1.0
    vec60 = (lm[1:] / scale).flatten()            # drop wrist → 20×3
    return label, vec60.tolist()

# ───────────────────────── main routine ──────────────────────────────────────
def vectorise_dataset(input_dir: Path, output_dir: Path,
                      show_progress: bool, chunk_pct: int = 7) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # collect paths
    img_iterable = list(iterate_images(input_dir)) if show_progress else iterate_images(input_dir)
    n_total = len(img_iterable) if show_progress else None
    if n_total == 0:
        raise RuntimeError("No *.jpg images found.")

    # pool settings
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    chunk = max(1, int(len(img_iterable) * chunk_pct / 100 / n_workers))
    print(f"[INFO] Using {n_workers} workers  |  chunk={chunk}")

    pbar = tqdm(total=n_total, desc="Vectorising", unit="img", file=sys.stdout) if show_progress else None

    fail_reasons = {"class_nothing": 0, "no_hand": 0, "corrupt_image": 0}
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for label, result in ex.map(one_image_to_vector, img_iterable, chunksize=chunk):
            if isinstance(result, list):  # good result
                    rows.append([label, *result])
            else:
                    fail_reasons[result] += 1
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()

    print(f"[INFO] Samples OK: {len(rows)}   |  failed/skipped: {sum(fail_reasons.values())}")
    for reason, count in fail_reasons.items():
        print(f"        - {reason:15}: {count}")

    # save global CSV
    header = ["label"] + [f"f{i}" for i in range(60)]
    full_csv = output_dir / "features_all.csv"
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(full_csv, index=False)
    print(f"[INFO] Saved → {full_csv}")

    # ------------------------------------------------------------------
    X, y = df.drop(columns="label").values, df["label"].values

    # ── SPLIT A ── 80 / 20 -------------------------------------------------------
    sss_80_20 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_a_idx, test_a_idx = next(sss_80_20.split(X, y))

    df.iloc[train_a_idx].to_csv(output_dir / "train80.csv", index=False)
    df.iloc[test_a_idx].to_csv(output_dir / "test20.csv",  index=False)
    print("[INFO] 80/20 split written: train80.csv  |  test20.csv")

    # ── SPLIT B ── 80 / 10 / 10 ---------------------------------------------------
    X_test, y_test =  df.iloc[test_a_idx].drop(columns="label").values,  df.iloc[test_a_idx]["label"].values

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_b_idx, test_b_idx = next(sss_test.split(X_test, y_test))


    df.iloc[val_b_idx]  .to_csv(output_dir / "val10.csv",   index=False)
    df.iloc[test_b_idx] .to_csv(output_dir / "test10.csv",  index=False)
    print("[INFO] 80/10/10 split written: train80.csv | val10.csv | test10.csv")


# ───────────────────────── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorise ASL dataset to CSV.")
    parser.add_argument("--input",  type=Path, default=Path("../data/filtered/clean"))
    parser.add_argument("--output", type=Path, default=Path("../data/processed"))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    vectorise_dataset(args.input, args.output, show_progress=not args.no_progress)
    print("[INFO] Done.")
