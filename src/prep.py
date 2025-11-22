import pandas as pd
import numpy as np

NUM_FEATURES = [
    # TODO: list main numeric columns after inspecting CSV in nb 01
]

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - drop duplicates, obvious identifiers
    - handle NA (median impute)
    - optional: clip extreme outliers (e.g., winsorize at 99th)
    """
    # TODO: implement after EDA decisions
    raise NotImplementedError

