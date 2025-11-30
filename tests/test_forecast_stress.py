import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.forecasting import forecast_with_scenario
from app.data_loader import load_processed

def test_large_horizon():
    df = load_processed()
    fut = forecast_with_scenario(df, future_days=120)
    assert len(fut) == 120

def test_negative_shift():
    df = load_processed()
    fut = forecast_with_scenario(df, future_days=10, discount_shift=-0.5)
    assert (fut["predicted_revenue"] >= 0).all()