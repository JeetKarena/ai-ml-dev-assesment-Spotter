"""Geographic route and operational interaction feature utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer route, haul categorization, and weight interaction features.

    Computes:
    - Route string (pickup to delivery)
    - Distance haul categories (short, medium, long)
    - Equipment and distance/weight combinations
    """
    res = df.copy()
    res["route"] = res["pickup"].astype(str) + " -> " + res["delivery"].astype(str)

    distance = pd.to_numeric(res["distance"], errors="coerce")
    res["distance_bucket"] = (
        pd.cut(distance, bins=[-np.inf, 250, 750, np.inf], labels=["short", "medium", "long"])
        .astype("object")
        .fillna("unknown")
    )

    res["equipment_x_distance"] = res["equipment"].astype(str) + "_" + res["distance_bucket"].astype(str)
    res["distance_per_weight"] = distance / res["weight"]
    res["weight_x_distance"] = res["weight"] * distance

    return res
