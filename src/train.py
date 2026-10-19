
from pathlib import Path
import subprocess
import sys

from data_loader import load_all_cities


def main():
    config_path = Path("config/data_paths.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config/data_paths.yaml bulunamadı. "
            "Lütfen config/data_paths.example.yaml dosyasını kopyalayıp doldur."
        )

    city_dfs, cfg = load_all_cities(config_path)
    datetime_col = "datetime"
    target_col = cfg["data"]["columns"]["target"]

    print("\n[INFO] Eğitim öncesi hızlı kontrol")
    for city, df in city_dfs.items():
        print(f"{city}: rows={len(df)}, cols={len(df.columns)}")
        print(f"  datetime min/max: {df[datetime_col].min()} / {df[datetime_col].max()}")
        print(f"  target null ratio: {df[target_col].isna().mean():.4f}")

    print("\n[INFO] Preprocess başlatılıyor...")
    subprocess.check_call([sys.executable, "src/preprocess.py"])
    print("[INFO] Preprocess tamamlandı. Sonraki adım: LSTM baseline.")


if __name__ == "__main__":
    main()
