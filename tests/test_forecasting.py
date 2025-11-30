import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
from app.forecasting import train_revenue_model, forecast_with_scenario, FEATURE_COLS
from app.data_loader import load_processed

MODEL_PATH = "models/revenue_model.pkl"

def test_train_revenue_model_creates_file():
    metrics = train_revenue_model()
    assert os.path.exists(MODEL_PATH)
    assert "MAE" in metrics
    assert "R2" in metrics

def test_forecast_scenario_output():
    df = load_processed()
    forecast_df = forecast_with_scenario(df, future_days=7)
    assert len(forecast_df) == 7
    assert "predicted_revenue" in forecast_df.columns

def test_features_exist():
    df = load_processed()
    for col in FEATURE_COLS:
        assert col in df.columns