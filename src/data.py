"""From the two DuckDB tables to one modelling frame per region.

    dataset("us")  ->  one row per month: the seven products, then every driver as it was
                       readable at that month end

Every driver observation carries two dates. period_end is the last day of the period the number
describes; available_at is the day it was published, period_end plus the source's lag. A quarter
of credit describing January to March does not exist until June and must not appear in an April
row, so the join is merge_asof backward on available_at: as of each month end, the most recent
value already published. Drivers become step functions rather than smooth lines, which costs a
tree nothing.
"""

import pandas as pd

from src import config as c, sql


def demand(region):
    """Seven products as columns, one row per month."""
    df = sql.query(f"SELECT product, month, kbd FROM demand WHERE region = '{region}'")
    df["month"] = pd.to_datetime(df["month"])
    return df.pivot(index="month", columns="product", values="kbd")


def predictors(region):
    """Each driver as a series at its own frequency, indexed by period_end."""
    df = sql.query(f"SELECT variable, period_end, value FROM predictors WHERE region = '{region}'")
    df["period_end"] = pd.to_datetime(df["period_end"])
    series = {name: s.set_index("period_end")["value"] for name, s in df.groupby("variable")}
    if region == "us":
        series["rpo_real"] = series.pop("brent") / series.pop("cpi")  # Brent in constant dollars
    return series


def dataset(region):
    """Demand joined to every driver published by each month end, on a complete monthly index so
    a gap in a source shows as a gap and not as a missing row."""
    df = demand(region)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS", name="month"))
    df = df.loc[c.SAMPLE[0]:c.SAMPLE[1]].reset_index()
    # the forecast origin: whatever was published by the last day of the month
    df["month_end"] = df["month"] + pd.offsets.MonthEnd(0)
    for name, s in predictors(region).items():
        s = s.dropna().rename(name).reset_index()
        s["available_at"] = s["period_end"] + pd.offsets.MonthEnd(c.LAG[name])
        s = s.sort_values("available_at")
        df = pd.merge_asof(df, s[["available_at", name]], left_on="month_end",
                           right_on="available_at", direction="backward",
                           tolerance=pd.Timedelta(days=c.STALE_DAYS))
        df = df.drop(columns="available_at")
    return df.drop(columns="month_end").set_index("month")
