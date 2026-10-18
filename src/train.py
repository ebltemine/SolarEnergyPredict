from pathlib import Path

from data_loader import load_all_cities


def main():
    config_path = Path("config/data_paths.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config/data_paths.yaml bulunamadı. "
            "Lütfen config/data_paths.example.yaml dosyasını kopyalayıp doldur."
        )

    city_dfs, cfg = load_all_cities(config_path)
    datetime_col = cfg["data"]["columns"]["datetime"]
    target_col = cfg["data"]["columns"]["target"]

    print("\n[INFO] Eğitim öncesi hızlı kontrol")
    for city, df in city_dfs.items():
        print(f"{city}: rows={len(df)}, cols={len(df.columns)}")
        print(f"  datetime min/max: {df[datetime_col].min()} / {df[datetime_col].max()}")
        print(f"  target null ratio: {df[target_col].isna().mean():.4f}")

    # TODO:
    # 1) preprocessing
    # 2) sequence creation
    # 3) baseline LSTM
    # 4) hybrid models
    # 5) evaluation & save metrics


if __name__ == "__main__":
    main()
