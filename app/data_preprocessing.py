import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data_loader import load_raw_transactions, build_daily_aggregates, save_processed

df = load_raw_transactions()
daily = build_daily_aggregates(df)
save_processed(daily)