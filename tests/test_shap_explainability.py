import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import shap
from app.forecasting import load_revenue_model, FEATURE_COLS
from app.data_loader import load_processed
import numpy as np

def test_shap_values_shape():
    df = load_processed()
    model = load_revenue_model()

    X = df[FEATURE_COLS].iloc[:50]  # small batch
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    assert len(shap_vals) == len(X)
    assert shap_vals.shape[1] == len(FEATURE_COLS)
    assert not np.isnan(shap_vals).any()