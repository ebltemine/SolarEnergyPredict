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


def _read_single_city_csv(
    file_path: str | Path,
    city_name: str,
    datetime_col: str,
    csv_params: dict
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"[{city_name}] CSV file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=csv_params.get("sep", ","),
        decimal=csv_params.get("decimal", "."),
        encoding=csv_params.get("encoding", "utf-8")
    )

    if datetime_col not in df.columns:
        raise ValueError(
            f"[{city_name}] datetime column '{datetime_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    df["city"] = city_name
    return df


def load_all_cities(config_path: str | Path) -> Tuple[Dict[str, pd.DataFrame], dict]:
    cfg = load_config(config_path)

    city_files = cfg["data"]["city_files"]
    datetime_col = cfg["data"]["columns"]["datetime"]
    target_col = cfg["data"]["columns"]["target"]
    csv_params = cfg["data"].get("csv", {})

    city_dfs: Dict[str, pd.DataFrame] = {}
    for city_name, file_path in city_files.items():
        df = _read_single_city_csv(
            file_path=file_path,
            city_name=city_name,
            datetime_col=datetime_col,
            csv_params=csv_params
        )

        if target_col not in df.columns:
            raise ValueError(
                f"[{city_name}] target column '{target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        city_dfs[city_name] = df

    return city_dfs, cfg


if __name__ == "__main__":
    # Örnek kullanım:
    # python src/data_loader.py
    config_file = "config/data_paths.yaml"  # local'de oluşturulacak
    city_dfs, cfg = load_all_cities(config_file)

    print("Loaded cities:")
    for city, df in city_dfs.items():
        print(f"- {city}: {df.shape}, date range: {df[cfg['data']['columns']['datetime']].min()} -> {df[cfg['data']['columns']['datetime']].max()}")
