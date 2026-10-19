
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_all_cities


def preprocess_city(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    # Gerekli kolonları seç
    cols = ["datetime", "city", target_col] + [c for c in feature_cols if c in df.columns]
    out = df[cols].copy()

    # Numeric dönüşüm
    for c in [target_col] + [c for c in feature_cols if c in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Zaman sıralama + duplicate datetime temizleme
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

    # Basit eksik değer tamamlama
    out = out.set_index("datetime")
    out = out.interpolate(method="time", limit_direction="both")
    out = out.fillna(method="ffill").fillna(method="bfill")
    out = out.reset_index()

    # Negatif target varsa 0'a kırp (GSI için mantıklı)
    out[target_col] = out[target_col].clip(lower=0)

    return out


def main():
    cfg_path = Path("config/data_paths.yaml")
    city_dfs, cfg = load_all_cities(cfg_path)

    target_col = cfg["data"]["columns"]["target"]
    feature_candidates = cfg["data"].get("feature_candidates", [])

    out_dir = Path("results/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for city, df in city_dfs.items():
        clean_df = preprocess_city(df, target_col=target_col, feature_cols=feature_candidates)

        out_file = out_dir / f"{city}_clean.csv"
        clean_df.to_csv(out_file, index=False, encoding="utf-8")

        summary_rows.append({
            "city": city,
            "rows": len(clean_df),
            "start": str(clean_df["datetime"].min()),
            "end": str(clean_df["datetime"].max()),
            "target_null_ratio": float(clean_df[target_col].isna().mean()),
            "target_mean": float(np.nanmean(clean_df[target_col])),
            "target_std": float(np.nanstd(clean_df[target_col])),
            "file": str(out_file),
        })

    summary = pd.DataFrame(summary_rows)
    summary_file = Path("results/processed/summary.csv")
    summary.to_csv(summary_file, index=False, encoding="utf-8")

    print("Preprocess tamamlandı.")
    print(summary)
    print(f"\nÖzet dosyası: {summary_file}")


if __name__ == "__main__":
    main()
