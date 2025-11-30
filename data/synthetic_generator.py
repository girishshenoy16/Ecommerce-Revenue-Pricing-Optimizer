import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

def generate_synthetic_transactions(
    start_date="2023-01-01",
    end_date="2023-12-31",
    n_products=20
):
    dates = pd.date_range(start_date, end_date, freq="D")
    categories = ["Electronics", "Fashion", "Grocery", "Home", "Beauty"]
    products = [f"P{i:03d}" for i in range(1, n_products + 1)]

    rows = []
    for date in dates:
        base_demand_multiplier = 1.0

        # Simple seasonality: weekends higher demand
        if date.weekday() >= 5:  # Sat, Sun
            base_demand_multiplier *= 1.3

        # Monthly seasonality (e.g., festive in Oct/Nov)
        if date.month in [10, 11]:
            base_demand_multiplier *= 1.4

        for prod in products:
            category = np.random.choice(categories)
            base_price = {
                "Electronics": np.random.uniform(2000, 6000),
                "Fashion":    np.random.uniform(500, 2000),
                "Grocery":    np.random.uniform(50, 500),
                "Home":       np.random.uniform(700, 3000),
                "Beauty":     np.random.uniform(200, 1200),
            }[category]

            # Discount percentage (0–40%)
            discount_pct = np.random.choice([0, 0, 0.1, 0.2, 0.3, 0.4])
            effective_price = base_price * (1 - discount_pct)

            # Demand = f(base_demand, price, randomness)
            # Higher discounts -> more units sold (simple elasticity)
            base_units = np.random.randint(5, 40)
            elasticity_effect = (1 + discount_pct * np.random.uniform(2, 4))
            category_noise = np.random.uniform(0.7, 1.3)

            units_sold = max(
                0,
                int(base_units * base_demand_multiplier * elasticity_effect * category_noise)
            )

            # Add some random noise including zero-sales days
            if np.random.rand() < 0.05:
                units_sold = 0

            revenue = units_sold * effective_price

            row = {
                "date": date,
                "product_id": prod,
                "category": category,
                "base_price": round(base_price, 2),
                "discount_pct": round(discount_pct, 2),
                "effective_price": round(effective_price, 2),
                "units_sold": units_sold,
                "revenue": round(revenue, 2),
                "is_promo": 1 if discount_pct > 0 else 0
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def main():
    df = generate_synthetic_transactions()
    out_path = RAW_DIR / "transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"[{datetime.now()}] Saved synthetic data to {out_path}")

if __name__ == "__main__":
    main()