import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.forecasting import train_revenue_model
from pathlib import Path

def test_visualizations_exist():
    train_revenue_model()

    report_dir = Path("reports/visuals")
    files = [
        "actual_vs_predicted.png",
        "residual_distribution.png",
        "feature_importance.png",
        "error_over_time.png",
    ]

    for f in files:
        assert (report_dir / f).exists(), f"Missing file: {f}"