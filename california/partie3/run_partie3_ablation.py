from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from partie3.multimodal_data import (
    load_aligned_multimodal,
    standardize_modalities_fit_transform,
    stratified_train_val_test_indices,
)
from partie3.train_ablation import run_ablation_suite


def main() -> None:
    project_root = PROJECT_ROOT
    out_dir = project_root / "runs" / "partie3_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_s2, M_s2, X_s1, X_static, y, _ = load_aligned_multimodal(project_root, n_classes=6)
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(y, test_size=0.15, val_size=0.15, random_state=42)
    (X_s2_n, X_s1_n, X_st_n), _ = standardize_modalities_fit_transform(X_s2, X_s1, X_static, idx_train)

    ablation = run_ablation_suite(
        X_s2_n,
        M_s2,
        X_s1_n,
        X_st_n,
        y,
        idx_train,
        idx_val,
        idx_test,
        n_classes=6,
        epochs=40,
        batch_size=64,
        random_seed=42,
    )

    rows = []
    for name, info in ablation.items():
        hist = info["history"].history
        best_idx = int(np.argmin(hist["val_loss"]))
        rows.append(
            {
                "config": name,
                "test_loss": info["test_loss"],
                "test_accuracy": info["test_accuracy"],
                "test_top2": info["test_top2"],
                "best_epoch": best_idx + 1,
                "best_val_loss": float(hist["val_loss"][best_idx]),
                "best_val_accuracy": float(hist["val_accuracy"][best_idx]),
            }
        )

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    out_csv = out_dir / "partie3_ablation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(df.round(4))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

