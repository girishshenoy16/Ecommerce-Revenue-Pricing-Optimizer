import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pricing import recommend_price, estimate_price_elasticity

def test_recommend_price_output():
    result = recommend_price(1000, "Electronics", target_change=0.1)
    assert "suggested_price" in result
    assert "elasticity_used" in result

def test_estimate_price_elasticity_runs():
    elasticity = estimate_price_elasticity()
    assert isinstance(elasticity, dict)
    # Should not be empty for synthetic dataset
    assert len(elasticity) > 0

def test_recommend_price_runs():
    res = recommend_price(current_price=1000.0, category="Electronics", target_change=0.1)
    assert "suggested_price" in res