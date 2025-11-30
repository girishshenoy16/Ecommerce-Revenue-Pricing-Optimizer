from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "transactions.csv"
MODELS_DIR = BASE_DIR / "models"
ELASTICITY_PATH = MODELS_DIR / "elasticity.json"


def estimate_price_elasticity():
    """
    Estimate price elasticity at category level using:
    log(units) = a + b * log(effective_price)
    elasticity = b
    """
    df = pd.read_csv(RAW_PATH)
    df = df[df["units_sold"] > 0].copy()

    df["log_units"] = np.log(df["units_sold"])
    df["log_price"] = np.log(df["effective_price"])

    elasticity = {}
    for category, group in df.groupby("category"):
        if group["log_price"].nunique() < 3:
            continue
        # Simple linear regression using numpy
        x = group["log_price"].values
        y = group["log_units"].values

        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        elasticity[category] = beta

    MODELS_DIR.mkdir(exist_ok=True)
    with open(ELASTICITY_PATH, "w") as f:
        json.dump(elasticity, f, indent=2)
    return elasticity


def load_elasticity():
    if not ELASTICITY_PATH.exists():
        return {}
    with open(ELASTICITY_PATH) as f:
        return json.load(f)


def recommend_price(
    current_price: float,
    category: str,
    target_change: float = 0.0,
    cost_price: float | None = None,
):
    """
    Uses elasticity to suggest a new price.

    target_change: desired percentage change in demand (e.g., +0.1 for +10% units)
    elasticity: %ΔQ / %ΔP  => %ΔP = %ΔQ / elasticity
    """
    elasticity = load_elasticity()
    e = elasticity.get(category, -1.0)  # assume negative

    if e == 0:
        e = -0.5  # avoid division by zero

    # %ΔP = %ΔQ / e
    pct_change_price = target_change / e
    new_price = current_price * (1 + pct_change_price)

    margin_info = None
    if cost_price is not None:
        old_margin = current_price - cost_price
        new_margin = new_price - cost_price
        margin_info = {
            "old_margin": old_margin,
            "new_margin": new_margin,
            "margin_change": new_margin - old_margin,
        }

    # Prevent negative or zero prices (needed for testing + business logic)
    new_price = max(new_price, 1.0)

    return {
        "elasticity_used": e,
        "suggested_price": round(float(new_price), 2),
        "pct_change_price": pct_change_price,
        "margin_info": margin_info,
    }