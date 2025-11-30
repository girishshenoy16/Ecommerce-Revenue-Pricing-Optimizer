import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.drift_utils import calculate_psi
import pandas as pd
import numpy as np

def test_psi_basic():
    a = pd.Series(np.random.normal(0, 1, 500))
    b = pd.Series(np.random.normal(0, 1, 500))

    psi = calculate_psi(a, b)
    assert psi < 0.1  # same distribution → small PSI

def test_psi_full_drift():
    a = pd.Series(np.random.normal(0, 1, 500))
    b = pd.Series(np.random.normal(10, 1, 500))

    psi = calculate_psi(a, b)
    assert psi > 0.5  # large drift