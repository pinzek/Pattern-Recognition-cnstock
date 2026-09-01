"""
Evaluation metrics for cross-sectional stock return prediction.

The functions in this file are intentionally lightweight and reusable. They are
designed for out-of-sample model evaluation, including IC, Rank IC, direction
accuracy, Sharpe ratio, and maximum drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _valid_xy(y_true, y_pred):
    """Return finite y_true and y_pred arrays."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def information_coefficient(y_true, y_pred) -> float:
    """
    Compute Pearson IC between realized returns and predicted scores.
    """

    y_true, y_pred = _valid_xy(y_true, y_pred)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rank_information_coefficient(y_true, y_pred) -> float:
    """
    Compute Spearman Rank IC between realized returns and predicted scores.
    """

    y_true, y_pred = _valid_xy(y_true, y_pred)

    if len(y_true) < 2:
        return np.nan

    result = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(result) if result is not None else np.nan


def direction_accuracy(y_true, y_pred) -> float:
    """
    Compute directional accuracy.

    A prediction is correct when predicted return and realized return have the
    same sign.
    """

    y_true, y_pred = _valid_xy(y_true, y_pred)

    if len(y_true) == 0:
        return np.nan

    return float(np.mean((y_true > 0) == (y_pred > 0)))


def mean_squared_error(y_true, y_pred) -> float:
    """
    Compute mean squared error.
    """

    y_true, y_pred = _valid_xy(y_true, y_pred)

    if len(y_true) == 0:
        return np.nan

    return float(np.mean((y_true - y_pred) ** 2))


def annualized_return(daily_returns, periods_per_year: int = 252) -> float:
    """
    Compute arithmetic annualized return from daily returns.
    """

    daily_returns = pd.Series(daily_returns).dropna()

    if daily_returns.empty:
        return np.nan

    return float(daily_returns.mean() * periods_per_year)


def annualized_volatility(daily_returns, periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility from daily returns.
    """

    daily_returns = pd.Series(daily_returns).dropna()

    if len(daily_returns) < 2:
        return np.nan

    return float(daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(daily_returns, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio without risk-free-rate adjustment.
    """

    daily_returns = pd.Series(daily_returns).dropna()

    if len(daily_returns) < 2:
        return np.nan

    vol = annualized_volatility(daily_returns, periods_per_year)

    if vol == 0 or not np.isfinite(vol):
        return np.nan

    return float(annualized_return(daily_returns, periods_per_year) / vol)


def max_drawdown(daily_returns) -> float:
    """
    Compute maximum drawdown from a daily return series.
    """

    daily_returns = pd.Series(daily_returns).fillna(0.0)

    if daily_returns.empty:
        return np.nan

    nav = (1.0 + daily_returns).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0

    return float(drawdown.min())


def daily_rank_ic(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    pred_col: str,
) -> pd.Series:
    """
    Compute daily cross-sectional Rank IC.

    Parameters
    ----------
    frame:
        Prediction result table.
    date_col:
        Date column name.
    target_col:
        Realized return column name.
    pred_col:
        Prediction score column name.
    """

    values = {}

    for date, group in frame.groupby(date_col):
        if group[target_col].notna().sum() < 2 or group[pred_col].notna().sum() < 2:
            values[date] = np.nan
        else:
            values[date] = rank_information_coefficient(
                group[target_col],
                group[pred_col],
            )

    return pd.Series(values, name="daily_rank_ic").sort_index()


def summarize_predictions(y_true, y_pred) -> dict:
    """
    Produce a compact prediction metric summary.
    """

    return {
        "IC": information_coefficient(y_true, y_pred),
        "Rank_IC": rank_information_coefficient(y_true, y_pred),
        "Direction_Accuracy": direction_accuracy(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "N": int(len(_valid_xy(y_true, y_pred)[0])),
    }
