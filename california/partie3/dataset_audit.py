from __future__ import annotations

from pathlib import Path

import numpy as np


def print_multimodal_audit(
    project_root: Path | str,
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    *,
    class_names: tuple[str, ...] = ("Grapes", "Rice", "Alfalfa", "Almonds", "Pistachios", "Others"),
) -> None:
    root = Path(project_root)
    raw_dir = root / "processed_data"
    x_files = [raw_dir / f"X_{s}.npy" for s in ("train", "val", "test")]
    if all(p.exists() for p in x_files):
        n_raw = int(sum(np.load(p, mmap_mode="r").shape[0] for p in x_files))
        n_now = int(X_s2.shape[0])
        print("--- Audit multimodal California ---")
        print(f"Processed S2 rows (raw): {n_raw}")
        print(f"After S1 ∩ static merge: {n_now}")
        print(f"Dropped rows (no match): {n_raw - n_now}")
    else:
        print("--- Audit multimodal California ---")
        print(f"Aligned samples: {X_s2.shape[0]}")

    print(f"\nShapes: S2 {X_s2.shape} | S1 {X_s1.shape} | static {X_static.shape}")
    print(f"NaN S2: {np.isnan(X_s2).sum()} | S1: {np.isnan(X_s1).sum()} | static: {np.isnan(X_static).sum()}")

    y = np.asarray(y).reshape(-1)
    print("\nClass distribution:")
    for c in range(len(class_names)):
        n = int(np.sum(y == c))
        pct = 100.0 * n / max(len(y), 1)
        print(f"  {c} {class_names[c]}: {n} ({pct:.1f}%)")


__all__ = ["print_multimodal_audit"]

