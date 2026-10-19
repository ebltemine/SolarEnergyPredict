
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def mape(y_true, y_pred):
    eps = 1e-8
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100


def train_eval_city(city_file: Path, city_name: str, target_col: str = "GSI", lookback: int = 24):
    df = pd.read_csv(city_file, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    base_features = ["SICAKLIK", "NISPI_NEM_%", "HIZI_m_sn", "AKTUEL_BASINC_hPa", "GSI60"]
    feature_cols = [c for c in base_features if c in df.columns]

    X_raw = df[feature_cols].astype(float).values
    y_raw = df[target_col].astype(float).values.reshape(-1, 1)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = X_raw[:train_end], y_raw[:train_end]
    X_val_raw, y_val_raw = X_raw[train_end:val_end], y_raw[train_end:val_end]
    X_test_raw, y_test_raw = X_raw[val_end:], y_raw[val_end:]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    Xtr, ytr = make_sequences(X_train, y_train, lookback)
    Xva, yva = make_sequences(X_val, y_val, lookback)
    Xte, yte = make_sequences(X_test, y_test, lookback)

    tf.keras.backend.clear_session()
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(Xtr.shape[1], Xtr.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=30,
        batch_size=64,
        verbose=0,
        callbacks=[early_stop]
    )

    y_pred_scaled = model.predict(Xte, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(yte).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    out_dir = Path("results/bilstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_datetimes = df.iloc[val_end:].reset_index(drop=True)["datetime"].iloc[lookback:].reset_index(drop=True)
    pd.DataFrame({
        "datetime": test_datetimes,
        "y_true": y_true,
        "y_pred": y_pred
    }).to_csv(out_dir / f"{city_name}_predictions.csv", index=False)

    pd.DataFrame(history.history).to_csv(out_dir / f"{city_name}_history.csv", index=False)
    model.save(out_dir / f"{city_name}_bilstm.keras")

    return {
        "city": city_name,
        "rows_total": n,
        "features": ",".join(feature_cols),
        "lookback": lookback,
        "mae": mae,
        "rmse": rmse,
        "mape": mape_val,
        "r2": r2
    }


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    processed_dir = Path("results/processed")
    files = {
        "ankara": processed_dir / "ankara_clean.csv",
        "bursa": processed_dir / "bursa_clean.csv",
        "canakkale": processed_dir / "canakkale_clean.csv",
    }

    rows = []
    for city, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"Processed file not found: {f}. Önce preprocess çalıştır.")
        print(f"[INFO] training city={city}")
        rows.append(train_eval_city(f, city_name=city, target_col="GSI", lookback=24))

    metrics = pd.DataFrame(rows).sort_values("rmse")
    out_file = Path("results/bilstm/metrics_bilstm.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_file, index=False)

    print("\\n=== BiLSTM Metrics ===")
    print(metrics)
    print(f"\\nKaydedildi: {out_file}")


if __name__ == "__main__":
    main()
