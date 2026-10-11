"""Direct multi-horizon XGBoost against the seasonal naive: from origin t, one model per step
forecasts demand(t + step), step = 1 ... 12.

The seasonal naive repeats the same calendar month of the latest year published at t. The
model predicts what it misses, y = log demand(t + step) - log naive, so a tree never has to
extrapolate a trend it has not seen, and beating the naive is the objective itself. The
forecast is the naive times exp(prediction).

Demand is published `lag` months late (2 for EIA, 1 for MPA), so the newest print at t is
d(t - lag); the drivers were aligned point-in-time in data.py. Nothing is scaled: a tree splits
on thresholds, and a monotone rescaling leaves every split where it is.

    python -m src.model    # the backtests, driver ablation and seed band, into results/
"""

import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config as c, data

HORIZON = 12

PARAMS = dict(objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=4, verbosity=0)
# eight fits on purpose: some 200 monthly origins, and a large search on 200 rows selects noise
GRID = [{"learning_rate": lr, "max_depth": d, "n_estimators": n}
        for lr in (0.03, 0.1) for d in (3, 5) for n in (200, 600)]


def back(step, lag):
    """Months from a target to its naive reference: the first multiple of 12 that reaches a
    month already published at the origin."""
    return 12 * math.ceil((step + lag) / 12)


def features(df, lag, step):
    """X at origin t, in year-on-year terms: the last twelve published demand prints and every
    driver as of t, each against its value a year earlier (degree days as a difference, they
    can be zero). y: log demand at t + step over the seasonal naive."""
    d = np.log(df["demand"])
    X = pd.DataFrame({f"yoy_lag{k}": d.shift(k) - d.shift(k + 12) for k in range(lag, lag + 12)})
    for name in df.columns.drop("demand"):
        if name.startswith("hdd"):
            X[name] = df[name] - df[name].shift(12)
        else:
            X[name] = np.log(df[name] / df[name].shift(12))
    return X, d.shift(-step) - d.shift(back(step, lag) - step)


def trainable(y, step, lag, origin):
    """Origins from TRAIN_START whose target had been published by the forecast origin, with
    neither the origin, the target nor its naive reference month in an excluded year."""
    target = y.index + pd.DateOffset(months=step)
    published = target + pd.DateOffset(months=lag) <= pd.Timestamp(origin)
    reference = target - pd.DateOffset(months=back(step, lag))
    excluded = (y.index.year.isin(c.EXCLUDE_YEARS) | target.year.isin(c.EXCLUDE_YEARS)
                | reference.year.isin(c.EXCLUDE_YEARS))
    return (y.index >= c.TRAIN_START) & published & y.notna() & ~excluded


def cv_score(X, y, step, lag, params, folds=3):
    """Expanding window on the trainable rows: each block is scored by a model fitted only on
    the rows whose target was published by the block's first origin, the real-time rule."""
    edges = [len(X) * k // (folds + 1) for k in range(folds + 2)]
    errors = []
    for a, b in zip(edges[1:-1], edges[2:]):
        fit = X.index + pd.DateOffset(months=step + lag) <= X.index[a]
        model = XGBRegressor(**PARAMS, **params).fit(X[fit], y[fit])
        errors.append(np.mean((y.iloc[a:b] - model.predict(X.iloc[a:b])) ** 2))
    return np.mean(errors)


def tune(df, lag, origin):
    """Parameters chosen at the longest step and reused for the shorter ones."""
    X, y = features(df, lag, HORIZON)
    rows = trainable(y, HORIZON, lag, origin)
    return min(GRID, key=lambda p: cv_score(X[rows], y[rows], HORIZON, lag, p))


def seasonal_naive(df, lag, origin):
    """Same calendar month in the latest year published at the origin: the free benchmark."""
    months = pd.date_range(pd.Timestamp(origin) + pd.DateOffset(months=1), periods=HORIZON,
                           freq="MS")
    reference = [m - pd.DateOffset(months=back(step, lag)) for step, m in enumerate(months, 1)]
    return pd.Series(df.loc[reference, "demand"].to_numpy(), index=months)


def forecast(df, lag, origin, params):
    """Demand at t+1 ... t+12 from origin t, one model per step."""
    origin, deviation = pd.Timestamp(origin), []
    for step in range(1, HORIZON + 1):
        X, y = features(df, lag, step)
        rows = trainable(y, step, lag, origin)
        fit = XGBRegressor(**PARAMS, **params).fit(X[rows], y[rows])
        deviation.append(fit.predict(X.loc[[origin]])[0])
    return seasonal_naive(df, lag, origin) * np.exp(deviation)


def backtest(df, lag, params=None, origins=None):
    """Forecast from every month end since BACKTEST_START. Parameters are tuned once, on what was
    published by the first origin, so the backtest years play no part in choosing them. Passing
    params and fewer origins reruns the same test cheaply, for the driver ablation."""
    if origins is None:
        origins = pd.date_range(c.BACKTEST_START, df["demand"].last_valid_index(), freq="MS")
    origins = origins[~origins.year.isin(c.EXCLUDE_YEARS)]
    params, rows = params or tune(df, lag, origins[0]), []
    for origin in origins:
        path = forecast(df, lag, origin, params)
        reference = [m - pd.DateOffset(months=back(s, lag)) for s, m in enumerate(path.index, 1)]
        rows.append(pd.DataFrame({
            "origin": origin, "step": range(1, HORIZON + 1), "reference": reference,
            "actual": df["demand"].reindex(path.index).to_numpy(),
            "xgboost": path.to_numpy(), "naive": seasonal_naive(df, lag, origin).to_numpy()},
            index=path.index.rename("month")))
    return pd.concat(rows), params


def score(bt):
    """MAPE by horizon, XGBoost against the seasonal naive, on target months already published
    with neither the target nor its naive reference in an excluded year."""
    bt = bt[bt["actual"].notna() & ~bt.index.year.isin(c.EXCLUDE_YEARS)
            & ~bt["reference"].dt.year.isin(c.EXCLUDE_YEARS)]
    ape = 100 * bt[["xgboost", "naive"]].sub(bt["actual"], axis=0).abs().div(bt["actual"], axis=0)
    return ape.groupby(bt["step"]).mean()


if __name__ == "__main__":
    # About 45 minutes, so notebook 02 reads the csv files instead. Rerun when the model changes.
    c.RESULTS.mkdir(exist_ok=True)
    tuned, ablation, seeds = {}, {}, {}
    for product, lag in c.DEMAND_LAG.items():
        print(product)
        df = data.dataset(product)
        bt, params = backtest(df, lag)
        bt.to_csv(c.RESULTS / f"backtest_{product}.csv")
        tuned[product] = params
        # one origin a quarter, every calendar season: the full backtest at a quarter of the cost.
        # Lags only, then one driver dropped at a time, same hyperparameters
        quarterly = pd.date_range(c.BACKTEST_START, df["demand"].last_valid_index(), freq="3MS")
        variants = {"all drivers": df, "lags only": df[["demand"]]}
        variants |= {f"without {x}": df.drop(columns=x) for x in df.columns.drop("demand")}
        for name, frame in variants.items():
            mape = score(backtest(frame, lag, params, quarterly)[0]).mean()
            ablation[(product, name)] = mape["xgboost"]
        ablation[(product, "seasonal naive")] = mape["naive"]
        # how much the score moves for no reason: same features, other seeds
        for seed in (1, 2, 3):
            PARAMS["random_state"] = seed
            mape = score(backtest(df, lag, params, quarterly)[0]).mean()
            seeds[(product, seed)] = mape["xgboost"]
        PARAMS["random_state"] = 42
    pd.DataFrame(tuned).T.to_csv(c.RESULTS / "params.csv")
    pd.Series(ablation).unstack(0).to_csv(c.RESULTS / "ablation.csv")
    pd.Series(seeds).unstack(0).to_csv(c.RESULTS / "seeds.csv")
