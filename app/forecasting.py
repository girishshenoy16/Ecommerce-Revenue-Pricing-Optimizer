import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "modeling_data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" 

MODEL_PATH = MODELS_DIR / "revenue_model.pkl"

VISUALS_DIR = REPORT_DIR / "visuals"
CSV_DIR = REPORT_DIR / "csv"


FEATURE_COLS = [
    "day_of_week",
    "week_of_year",
    "month",
    "avg_discount",
    "promo_share",
    "total_units",
]

TARGET_COL = "total_revenue"


def train_revenue_model():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)


    joblib.dump(model, MODEL_PATH)

    # ============================
    # 1. Actual vs Predicted
    # ============================
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"].iloc[-len(y_test):], y_test, label="Actual", linewidth=2)
    plt.plot(df["date"].iloc[-len(y_test):], y_pred, label="Predicted", linestyle="--")
    plt.title("Actual vs Predicted Revenue (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out1 = VISUALS_DIR / "actual_vs_predicted.png"
    plt.savefig(out1)
    plt.close()

    # ============================
    # 2. Residual Distribution
    # ============================
    residuals = y_test - y_pred
    
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("Residual Distribution (Test Set)")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    out2 = VISUALS_DIR / "residual_distribution.png"
    plt.savefig(out2)
    plt.close()

    # ============================
    # 3. Feature Importance
    # ============================
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    features_sorted = np.array(FEATURE_COLS)[sorted_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(features_sorted, importances[sorted_idx])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()

    out3 = VISUALS_DIR / "feature_importance.png"
    plt.savefig(out3)
    plt.close()

    # ============================
    # 4. Error Over Time
    # ============================
    error = np.abs(y_test - y_pred)

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"].iloc[-len(error):], error, color="red")
    plt.title("Prediction Error Over Time (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()

    out4 = VISUALS_DIR / "error_over_time.png"
    plt.savefig(out4)
    plt.close()


    metrics = { 
        "MAE": mae, 
        "RMSE": rmse, 
        "R2": r2,
        "MAPE": mape
    }

    metrics_path = CSV_DIR / "evaluation_report.txt"
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved evaluation report → {metrics_path}")

    return metrics


def load_revenue_model():
    return joblib.load(MODEL_PATH)


def forecast_with_scenario(
    daily_df: pd.DataFrame,
    future_days: int = 14,
    discount_shift: float = 0.0,
    promo_shift: float = 0.0,
):
    """
    Very simple scenario forecast:
    - Use last known features, adjust discount & promo, predict future revenue.
    """
    model = load_revenue_model()
    df = daily_df.copy().sort_values("date")

    last_row = df.iloc[-1]
    future_rows = []
    current_date = last_row["date"]

    for i in range(1, future_days + 1):
        current_date = current_date + pd.Timedelta(days=1)

        row = {
            "date": current_date,
            "day_of_week": current_date.weekday(),
            "week_of_year": current_date.isocalendar().week,
            "month": current_date.month,

            # Scenario adjustments
            "avg_discount": max(0.0, min(0.6, last_row["avg_discount"] + discount_shift)),
            "promo_share": max(0.0, min(1.0, last_row["promo_share"] + promo_shift)),

            # Assume units follow last known (simplified)
            "total_units": last_row["total_units"],
        }
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    X_future = future_df[FEATURE_COLS]
    future_df["predicted_revenue"] = model.predict(X_future)
    return future_df