import pandas as pd
import numpy as np

NUM_FEATURES = [
    'BALANCE',
    'BALANCE_FREQUENCY',
    'PURCHASES',
    'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE',
    'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY',
    'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY',
    'CASH_ADVANCE_TRX',
    'PURCHASES_TRX',
    'CREDIT_LIMIT',
    'PAYMENTS',
    'MINIMUM_PAYMENTS',
    'PRC_FULL_PAYMENT',
    'TENURE',
]

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - drop duplicates, obvious identifiers
    - handle NA (median impute)
    - optional: clip extreme outliers (e.g., winsorize at 99th)
    """
    # TODO: implement after EDA decisions
    df = df.drop_duplicates()
    df = clip_outliers(df, NUM_FEATURES, percentile=0.99)
    df = log1p_skewed(df, NUM_FEATURES)
    for col in NUM_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    return df

def clip_outliers(df: pd.DataFrame, cols: list[str], percentile: float = 0.99) -> pd.DataFrame:
    """
    Clip outliers at a given percentile
    """
    for col in cols:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(percentile))
    return df

def log1p_skewed(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply log1p transformation to skewed columns
    """
    for col in cols:
        df[col] = np.log1p(df[col])
    return df