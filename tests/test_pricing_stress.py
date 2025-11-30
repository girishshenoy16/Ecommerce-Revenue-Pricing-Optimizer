import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from app.pricing import recommend_price

def test_zero_cost():
    res = recommend_price(100, "Electronics", cost_price=0)
    assert "suggested_price" in res

def test_negative_elasticity():
    res = recommend_price(100, "Electronics", target_change=-0.5)
    assert res["suggested_price"] > 0

def test_high_change():
    res = recommend_price(100, "Electronics", target_change=2)
    assert res["suggested_price"] > 0