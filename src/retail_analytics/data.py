"""Synthetic retail transactions generator and loader.

The generator produces realistic-looking data (growth trend, seasonality,
repeat purchasing, regional/channel mix) so the project runs out of the box.
Swap in a real dataset (e.g. UCI Online Retail) via `load_transactions`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    "Electronics": (80, 600),
    "Home & Kitchen": (15, 150),
    "Apparel": (10, 90),
    "Beauty": (5, 60),
    "Sports": (12, 200),
    "Books": (4, 30),
}
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Web", "Mobile App", "Store"]
REQUIRED_COLUMNS = {
    "order_id", "customer_id", "order_date", "category",
    "quantity", "unit_price", "region", "channel",
}


def generate_transactions(n_customers: int = 2000, seed: int = 42,
                          start: str = "2023-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    signup_idx = np.clip(rng.exponential(scale=len(days) / 3, size=n_customers).astype(int), 0, len(days) - 1)
    # Each customer has a purchase propensity (orders per year) and a home region.
    propensity = rng.gamma(shape=1.5, scale=2.0, size=n_customers) + 0.3
    region = rng.choice(REGIONS, size=n_customers, p=[0.3, 0.25, 0.2, 0.25])

    rows, order_id = [], 100000
    cats = list(CATEGORIES)
    for cid in range(n_customers):
        first = signup_idx[cid]
        n_orders = max(1, rng.poisson(propensity[cid] * (len(days) - first) / 365))
        offsets = np.sort(rng.integers(first, len(days), size=n_orders))
        offsets[0] = first
        for k, off in enumerate(offsets):
            d = days[off]
            # Seasonality: Nov/Dec spike, mild summer dip (first order always kept).
            season = 1.0 + 0.5 * (d.month in (11, 12)) - 0.1 * (d.month in (6, 7))
            if k > 0 and rng.random() > min(1.0, season / 1.5 + 0.35):
                continue
            order_id += 1
            for _ in range(rng.integers(1, 4)):
                cat = rng.choice(cats, p=[0.15, 0.2, 0.22, 0.15, 0.13, 0.15])
                lo, hi = CATEGORIES[cat]
                rows.append((
                    order_id, f"C{cid:05d}", d, cat,
                    int(rng.integers(1, 4)),
                    round(float(rng.uniform(lo, hi)), 2),
                    region[cid],
                    rng.choice(CHANNELS, p=[0.5, 0.3, 0.2]),
                ))
    df = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date", "category",
                                     "quantity", "unit_price", "region", "channel"])
    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def load_transactions(path: str) -> pd.DataFrame:
    """Load a CSV of transactions and validate the schema."""
    df = pd.read_csv(path, parse_dates=["order_date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return clean_transactions(df)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows, remove duplicates and add a revenue column."""
    df = df.drop_duplicates().dropna(subset=["order_id", "customer_id", "order_date"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)].copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df.reset_index(drop=True)
