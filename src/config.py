"""Paths, the three products, and every publication-lag assumption."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "oil.duckdb"
RESULTS = ROOT / "results"  # backtests written by python -m src.model, read by notebook 02

# raw files, one folder per product
GASOLINE = DATA / "US_gasoline"
DISTILLATE = DATA / "US_distillate_fuel_oil"
BUNKER = DATA / "SGP_bunker_fuel_oil"

# --- when a number becomes readable -------------------------------------------
# Months from the end of the period a number describes to the day it is published, from each
# source's release calendar: available_at = period_end + LAG month ends.
# The targets: EIA's Petroleum Supply Monthly comes out at the end of the second month after,
# MPA's bunker sales in the middle of the following month.
DEMAND_LAG = {"gasoline": 2, "distillate": 2, "bunker": 1}
LAG = {
    "real_price": 1,       # EIA retail price over the CPI, which BLS publishes mid-month after
    "vmt": 2,              # FHWA Traffic Volume Trends: August 2026 came out on 7 October
    "payrolls": 1,         # BLS Employment Situation, first Friday of the following month
    "inc": 1,              # BEA personal income, end of the following month
    "ind": 1,              # Fed G.17, mid following month
    "ind_manuf": 1,
    "truck": 3,            # ATA tonnage via BTS: July 2026 came out on 1 October
    "hdd": 1,              # NOAA CPC monthly summary, first days of the following month
    "hdd_oil": 1,
    "bunker_calls": 1,     # MPA port statistics, published with the bunker sales
    "bunker_calls_gt": 1,
    "cargo": 1,
    "tankers": 1,
    "wtr": 2,              # CPB, around the 25th of the second month after
    "brent": 0,            # a monthly average of daily prices, known at month end
}
# EIA's Weekly Petroleum Status Report: the Wednesday after the Friday that ends the week,
# Thursday in holiday weeks
WEEKLY_LAG_DAYS = 6

# Past this a value is stale rather than late: it goes missing, which XGBoost handles and which
# is honest about a series that has stopped publishing.
STALE_DAYS = 730

# --- the sample and the experiment ---------------------------------------------
# From 2000 every driver exists, except MPA's tanker arrivals (2002)
SAMPLE_START = "2000-01-01"
# Training rows start in 2001, the first year with a year-earlier value for the year-on-year
# features and the naive reference. Starting in 2010 instead cost the US products their edge.
TRAIN_START = "2001-01-01"
# The backtest forecasts from every month end since 2015, fourteen years of training data in.
BACKTEST_START = "2015-01-01"
# 2020 is a shock, not a pattern to learn: no origin, target or naive reference month in it.
EXCLUDE_YEARS = [2020]
