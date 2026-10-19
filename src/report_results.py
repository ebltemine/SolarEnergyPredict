
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    comparison_path = Path("results/model_comparison.csv")
    if not comparison_path.exists():
        raise FileNotFoundError("results/model_comparison.csv bulunamadı.")

    df = pd.read_csv(comparison_path)

    out_dir = Path("results/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Şehir bazlı en iyi model (RMSE min)
    best_per_city = (
        df.sort_values(["city", "rmse"])
          .groupby("city", as_index=False)
          .first()
          .sort_values("city")
    )
    best_per_city.to_csv(out_dir / "best_model_per_city.csv", index=False, encoding="utf-8")

    # 2) Genel ortalama metrikler (model bazında)
    avg_metrics = (
        df.groupby("model", as_index=False)[["mae", "rmse", "mape", "r2"]]
          .mean()
          .sort_values("rmse")
    )
    avg_metrics.to_csv(out_dir / "average_metrics_by_model.csv", index=False, encoding="utf-8")

    # 3) Pivot tablo (tez için kullanışlı)
    pivot_rmse = df.pivot_table(index="city", columns="model", values="rmse")
    pivot_mae = df.pivot_table(index="city", columns="model", values="mae")
    pivot_mape = df.pivot_table(index="city", columns="model", values="mape")
    pivot_r2 = df.pivot_table(index="city", columns="model", values="r2")

    pivot_rmse.to_csv(out_dir / "pivot_rmse.csv", encoding="utf-8")
    pivot_mae.to_csv(out_dir / "pivot_mae.csv", encoding="utf-8")
    pivot_mape.to_csv(out_dir / "pivot_mape.csv", encoding="utf-8")
    pivot_r2.to_csv(out_dir / "pivot_r2.csv", encoding="utf-8")

    # 4) Grafikler
    plt.style.use("seaborn-v0_8-whitegrid")

    # 4.a Ortalama RMSE bar chart
    plt.figure(figsize=(8, 5))
    x = avg_metrics["model"]
    y = avg_metrics["rmse"]
    plt.bar(x, y)
    plt.title("Average RMSE by Model")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    for i, v in enumerate(y):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_rmse_by_model.png", dpi=150)
    plt.close()

    # 4.b Şehir bazlı RMSE karşılaştırma
    rmse_plot = df.pivot_table(index="city", columns="model", values="rmse")
    rmse_plot.plot(kind="bar", figsize=(10, 6))
    plt.title("RMSE Comparison by City and Model")
    plt.xlabel("City")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_by_city_model.png", dpi=150)
    plt.close()

    # 4.c Şehir bazlı R2 karşılaştırma
    r2_plot = df.pivot_table(index="city", columns="model", values="r2")
    r2_plot.plot(kind="bar", figsize=(10, 6))
    plt.title("R2 Comparison by City and Model")
    plt.xlabel("City")
    plt.ylabel("R2")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "r2_by_city_model.png", dpi=150)
    plt.close()

    print("Raporlar üretildi:", out_dir)
    print("\n[Best Model Per City]")
    print(best_per_city)
    print("\n[Average Metrics By Model]")
    print(avg_metrics)


if __name__ == "__main__":
    main()
