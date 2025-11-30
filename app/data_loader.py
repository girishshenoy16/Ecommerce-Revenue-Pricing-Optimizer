from pathlib import Path
from app.cleaning import clean_transactions
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "transactions.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "modeling_data.csv"

def load_raw_transactions():
    return pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = clean_transactions(df)
    return df


def build_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date")
        .agg(
            total_revenue=("revenue", "sum"),
            total_units=("units_sold", "sum"),
            avg_discount=("discount_pct", "mean"),
            promo_share=("is_promo", "mean"),
        )
        .reset_index()
    )
    daily["day_of_week"] = daily["date"].dt.weekday
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["month"] = daily["date"].dt.month
    return daily

def save_processed(daily: pd.DataFrame):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_PATH, index=False)

def load_processed():
    return pd.read_csv(PROCESSED_PATH, parse_dates=["date"])