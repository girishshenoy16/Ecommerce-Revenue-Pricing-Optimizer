import pandas as pd

def top_categories_by_revenue(df: pd.DataFrame, top_n: int = 5):
    out = (
        df.groupby("category")
        .agg(total_revenue=("revenue", "sum"), units=("units_sold", "sum"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(top_n)
    )

    return out

def promo_effect(df: pd.DataFrame):
    grouped = (
        df.groupby("is_promo")
        .agg(avg_units=("units_sold", "mean"), avg_revenue=("revenue", "mean"))
        .reset_index()
    )
    
    return grouped

def daily_summary(daily_df: pd.DataFrame):
    stats = {
        "total_days": len(daily_df),
        "avg_daily_revenue": daily_df["total_revenue"].mean(),
        "max_daily_revenue": daily_df["total_revenue"].max(),
        "min_daily_revenue": daily_df["total_revenue"].min(),
    }

    return stats

def data_quality_report(df: pd.DataFrame):
    return {
        "rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "min_date": str(df["date"].min()),
        "max_date": str(df["date"].max()),
        "num_categories": df["category"].nunique()
    }
