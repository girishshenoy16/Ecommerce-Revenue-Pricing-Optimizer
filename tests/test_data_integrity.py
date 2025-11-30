import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data_loader import load_raw_transactions, load_processed

def test_no_missing_values():
    df = load_processed()
    assert df.isna().sum().sum() == 0

def test_valid_ranges():
    df = load_processed()
    assert (df["avg_discount"] >= 0).all()
    assert (df["promo_share"] >= 0).all()
    assert (df["total_units"] >= 0).all()
    assert (df["total_revenue"] >= 0).all()