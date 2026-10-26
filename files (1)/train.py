"""Freight rate prediction: chronological validation, final fit, and prediction files.

Run:  python train.py            (reads ./data, writes validation_predictions.csv,
                                  data/december_chart_inputs.csv (completed), metrics.json)
Then: python score.py --predictions validation_predictions.csv \
                      --december-predictions data/december_chart_inputs.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

DATA = Path("data")
SEED = 0
EQUIPMENT = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
FEATURES = [
    "log_dist", "distance", "weight_abs", "weight_missing", "market_idx", "market_idx_missing",
    "quote_signal", "quote_dev", "equipment_code", "dist_ratio",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_features(frame: pd.DataFrame, quote_center: float) -> pd.DataFrame:
    """Row-level cleaning + features. Uses no target information."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Data quality: ~0.6% of weights are negative (sign error; magnitudes look valid) -> abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_abs"] = out["weight"].abs()          # NaN stays NaN; trees handle it natively
    out["market_idx_missing"] = out["market_index"].isna().astype(int)
    out["market_idx"] = out["market_index"]          # imputed per day in impute_market_index()
    out["log_dist"] = np.log(out["distance"])
    hav = haversine_miles(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
    out["dist_ratio"] = out["distance"] / hav        # >1.5 flags odd distance/coordinate pairs
    out["equipment_code"] = out["equipment"].map(EQUIPMENT)
    out["quote_dev"] = (out["quote_signal"] - quote_center).abs()  # rate vs quote_signal is U-shaped
    return out


def impute_market_index(frame: pd.DataFrame, daily_mean: pd.Series) -> pd.DataFrame:
    """market_index is a market-wide daily signal (within-day std ~0.025), so fill gaps with that
    day's mean across all loads. Feature-only, no target leakage."""
    frame["market_idx"] = frame["market_index"].fillna(frame["date"].map(daily_mean))
    return frame


def make_model() -> HistGradientBoostingRegressor:
    # absolute_error (median) loss is robust to the ~1.4% extreme-rate outliers in the labels
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
    )


def metrics(actual, pred) -> dict:
    err = pred - actual
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_pct": float((np.abs(err) / actual).mean() * 100),
        "MedAPE_pct": float(np.median(np.abs(err) / actual) * 100),
    }


def main() -> None:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

    quote_center = float(train["quote_signal"].median())
    # daily market-index means come from feature columns of every file (train + validation)
    daily_mean = pd.concat([train[["date", "market_index"]], valid[["date", "market_index"]]]) \
        .groupby("date")["market_index"].mean()
    daily_quote = valid.groupby("date")["quote_signal"].mean()

    tr = impute_market_index(build_features(train, quote_center), daily_mean)
    tr["target"] = np.log(tr["posted_rate"] / tr["distance"])  # model rate-per-mile in log space

    # ---- Chronological validation: expanding window, always predict the NEXT 2 months ----
    folds = [("2025-07-01", "2025-09-01"), ("2025-08-01", "2025-10-01"), ("2025-09-01", "2025-11-01")]
    rows = []
    for start, end in folds:
        fit_part, test_part = tr[tr.date < start], tr[(tr.date >= start) & (tr.date < end)]
        model = make_model().fit(fit_part[FEATURES], fit_part["target"])
        pred = np.exp(model.predict(test_part[FEATURES])) * test_part["distance"]
        base_rpm = (fit_part.posted_rate / fit_part.distance).groupby(fit_part.equipment).median()
        base = test_part["equipment"].map(base_rpm) * test_part["distance"]
        rows.append({"fold": f"train<{start} -> test {start}..{end}",
                     **{f"model_{k}": v for k, v in metrics(test_part.posted_rate, pred).items()},
                     **{f"baseline_{k}": v for k, v in metrics(test_part.posted_rate, base).items()}})
    cv = pd.DataFrame(rows)
    print(cv.round(3).to_string(index=False))

    # ---- Unseen-city check (validation set has 8 cities absent from training) ----
    cities = sorted(set(train.pickup) | set(train.delivery))
    held = set(np.random.RandomState(1).choice(cities, 8, replace=False))
    mask = tr.pickup.isin(held) | tr.delivery.isin(held)
    m = make_model().fit(tr[~mask][FEATURES], tr[~mask]["target"])
    p = np.exp(m.predict(tr[mask][FEATURES])) * tr[mask]["distance"]
    unseen = metrics(tr[mask].posted_rate, p)
    print("unseen-city holdout:", {k: round(v, 3) for k, v in unseen.items()})

    # ---- Permutation importance on the last chronological fold ----
    fit_part, test_part = tr[tr.date < folds[-1][0]], tr[(tr.date >= folds[-1][0]) & (tr.date < folds[-1][1])]
    m = make_model().fit(fit_part[FEATURES], fit_part["target"])
    pi = permutation_importance(m, test_part[FEATURES], test_part["target"], n_repeats=3,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    importance = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    print(importance.round(4).to_string())

    # ---- Final fit on ALL labeled data, predict the 12,000 validation loads ----
    final = make_model().fit(tr[FEATURES], tr["target"])
    va = impute_market_index(build_features(valid, quote_center), daily_mean)
    va["predicted_rate"] = np.exp(final.predict(va[FEATURES])) * va["distance"]
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(va[["load_id", "predicted_rate"]], on="load_id", how="left")
    out["predicted_rate"] = out["predicted_rate"].round(2)
    assert out.predicted_rate.notna().all() and len(out) == 12_000
    out.to_csv("validation_predictions.csv", index=False)

    # ---- December chart: only the date varies. Inputs missing from the chart file are filled
    # with the city-pair's fixed coordinates (from training) and that day's market-wide signals.
    coords = {}
    for city in ("Lexington", "Fort Wayne"):
        r = train[train.pickup == city].iloc[0] if (train.pickup == city).any() else train[train.delivery == city].iloc[0]
        col = "pickup" if r.pickup == city else "delivery"
        coords[city] = (r[f"{col}_lat"], r[f"{col}_lon"])
    d = dec.copy()
    d["pickup_lat"], d["pickup_lon"] = coords["Lexington"]
    d["delivery_lat"], d["delivery_lon"] = coords["Fort Wayne"]
    d["market_index"] = d["date"].map(daily_mean)
    d["quote_signal"] = d["date"].map(daily_quote)
    d = impute_market_index(build_features(d, quote_center), daily_mean)
    dec["predicted_rate"] = (np.exp(final.predict(d[FEATURES])) * d["distance"]).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)

    json.dump({"cv": cv.to_dict("records"), "unseen_city": unseen,
               "importance": importance.round(5).to_dict()}, open("metrics.json", "w"), indent=2)
    print("wrote validation_predictions.csv and completed data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
