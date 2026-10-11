"""From the two DuckDB tables to one modelling frame per product.

    dataset("gasoline")  ->  one row per month: demand, then every driver as it was readable at
                             that month end

Every driver observation carries two dates. period_end is the last day of the period the number
describes; available_at is the day it was published, period_end plus the source's lag. Truck
tonnage for July is not out until October and must not appear in an August or September row,
so the join is merge_asof backward on available_at: as of each month end, the most recent value
already published. Drivers become step functions rather than smooth lines, which costs a tree
nothing.
"""

import pandas as pd

from src import config as c, sql


def demand(product):
    """The target, indexed by the first day of its month."""
    df = sql.query(f"SELECT date_trunc('month', period_end) AS month, value AS demand "
                   f"FROM demand WHERE product = '{product}' ORDER BY month")
    df["month"] = pd.to_datetime(df["month"])
    return df.set_index("month")


def predictors(product):
    """Each driver as a series at its own frequency, indexed by period_end. Exports and stocks
    stay in the database as supply-side context and are not fed to the model."""
    df = sql.query(f"SELECT variable, period_end, value FROM predictors "
                   f"WHERE product = '{product}' AND variable NOT IN ('exports', 'stocks')")
    df["period_end"] = pd.to_datetime(df["period_end"])
    series = {name: s.set_index("period_end")["value"].sort_index()
              for name, s in df.groupby("variable")}
    if product in ("gasoline", "distillate"):
        # USD/gal in 1982-84 dollars, the CPI base
        series["real_price"] = series.pop("retail_price") / series.pop("cpi") * 100
        # one week is noisy: EIA itself reports the four-week average
        series["product_supplied_4wk"] = series.pop("product_supplied_weekly").rolling("28D").mean()
    return series


def dataset(product):
    """Demand joined to every driver published by each month end, on a complete monthly index so
    a gap in a source shows as a gap and not as a missing row. The index runs to the latest
    origin: the month end by which the last demand figure was published."""
    df = demand(product)
    last_origin = df.index.max() + pd.DateOffset(months=c.DEMAND_LAG[product])
    df = df.reindex(pd.date_range(c.SAMPLE_START, last_origin, freq="MS", name="month"))
    df = df.reset_index()
    # the forecast origin: whatever was published by the last day of the month
    df["month_end"] = df["month"] + pd.offsets.MonthEnd(0)
    for name, s in predictors(product).items():
        s = s.dropna().rename(name).reset_index()
        if name == "product_supplied_4wk":
            s["available_at"] = s["period_end"] + pd.Timedelta(days=c.WEEKLY_LAG_DAYS)
        else:
            s["available_at"] = s["period_end"] + pd.offsets.MonthEnd(c.LAG[name])
        s = s.sort_values("available_at")
        df = pd.merge_asof(df, s[["available_at", name]], left_on="month_end",
                           right_on="available_at", direction="backward",
                           tolerance=pd.Timedelta(days=c.STALE_DAYS))
        df = df.drop(columns="available_at")
    return df.drop(columns="month_end").set_index("month")
