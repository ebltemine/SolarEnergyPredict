from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_datetime_from_parts(df: pd.DataFrame) -> pd.Series:
    required = ["YIL", "AY", "GUN", "SAAT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Datetime parçaları eksik: {missing}. Beklenen kolonlar: {required}")

    tmp = pd.DataFrame({
        "year": pd.to_numeric(df["YIL"], errors="coerce"),
        "month": pd.to_numeric(df["AY"], errors="coerce"),
        "day": pd.to_numeric(df["GUN"], errors="coerce"),
        "hour": pd.to_numeric(df["SAAT"], errors="coerce"),
    })

    # SAAT bazen 24 gelebilir, 0'a çekiyoruz
    tmp["hour"] = tmp["hour"].fillna(0).clip(lower=0, upper=24)
    tmp.loc[tmp["hour"] == 24, "hour"] = 0

    return pd.to_datetime(tmp, errors="coerce")


def _read_single_city_csv(
    file_path: str | Path,
    city_name: str,
    csv_params: dict
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"[{city_name}] CSV file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=csv_params.get("sep", ","),
        decimal=csv_params.get("decimal", "."),
        encoding=csv_params.get("encoding", "utf-8"),
    )

    df["datetime"] = _build_datetime_from_parts(df)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["city"] = city_name
    return df


def load_all_cities(config_path: str | Path) -> Tuple[Dict[str, pd.DataFrame], dict]:
    cfg = load_config(config_path)

    city_files = cfg["data"]["city_files"]
    target_col = cfg["data"]["columns"]["target"]
    csv_params = cfg["data"].get("csv", {})

    city_dfs: Dict[str, pd.DataFrame] = {}
    for city_name, file_path in city_files.items():
        df = _read_single_city_csv(
            file_path=file_path,
            city_name=city_name,
            csv_params=csv_params
        )

        if target_col not in df.columns:
            raise ValueError(
                f"[{city_name}] target column '{target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        city_dfs[city_name] = df

    return city_dfs, cfg
