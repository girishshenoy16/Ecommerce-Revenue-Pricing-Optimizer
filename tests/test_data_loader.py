import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data_loader import load_raw_transactions, load_processed

def test_raw_data_loads():
    df = load_raw_transactions()
    assert len(df) > 0
    assert "product_id" in df.columns
    assert "revenue" in df.columns

def test_processed_data_loads():
    df = load_processed()
    assert len(df) > 0
    assert "total_revenue" in df.columns
    assert "day_of_week" in df.columns