import pandas as pd
import numpy as np

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ecommerce transactional dataset.
    """

    df = df.copy()

    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 2. Fix dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notnull()]

    # 3. Fix numeric columns
    numeric_cols = ["base_price", "discount_pct", "effective_price", "units_sold", "revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    # 4. Fix discounts (valid range 0–0.6)
    df["discount_pct"] = df["discount_pct"].clip(0, 0.6)

    # 5. Recompute effective price
    df["effective_price"] = df["base_price"] * (1 - df["discount_pct"])

    # 6. Remove negative units_sold
    df.loc[df["units_sold"] < 0, "units_sold"] = 0
    df["units_sold"] = df["units_sold"].astype(int)

    # 7. Remove extreme unit outliers using IQR
    Q1 = df["units_sold"].quantile(0.25)
    Q3 = df["units_sold"].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    df["units_sold"] = df["units_sold"].clip(lower=0, upper=upper_limit)

    # 8. Recalculate revenue
    df["revenue"] = df["units_sold"] * df["effective_price"]

    # 9. Recalculate promo flag
    df["is_promo"] = (df["discount_pct"] > 0).astype(int)

    return df