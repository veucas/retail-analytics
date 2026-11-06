import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retail_analytics import data


def test_clean_removes_invalid_rows_and_adds_revenue():
    df = pd.DataFrame({
        "order_id": [1, 1, 2, 3], "customer_id": ["a", "a", "b", "c"],
        "order_date": pd.to_datetime(["2024-01-01"] * 4), "category": ["x"] * 4,
        "quantity": [1, 1, -2, 2], "unit_price": [10.0, 10.0, 5.0, 4.0],
        "region": ["N"] * 4, "channel": ["Web"] * 4,
    })
    out = data.clean_transactions(df)
    assert len(out) == 2
    assert out["revenue"].tolist() == [10.0, 8.0]
