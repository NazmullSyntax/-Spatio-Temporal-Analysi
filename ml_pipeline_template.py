"""
Dhaka Green Space / Urban Heat project - analysis pipeline template.

This is a SCAFFOLD, not a finished analysis: every function assumes you
have already produced the inputs listed in its docstring (from the GEE
script and the tabular data-collection phase). Nothing here invents
numbers - if an input file is missing, the functions raise/skip with a
clear message rather than substituting fabricated values.

Expected input files (see Reproducibility Plan / folder structure in
the main proposal document):
    03_tabular/ward_year_panel.csv
        columns: year, ward_id, ndvi_mean, ndbi_mean, lst_day_mean,
                 lst_night_mean, green_pct, builtup_pct,
                 population_density, pm25 (optional), health_index (optional),
                 economic_index (optional), data_resolution (ward/city/national)

Install once: pip install pandas numpy scikit-learn xgboost statsmodels
              matplotlib seaborn geopandas rasterio --break-system-packages
"""

import os
import numpy as np
import pandas as pd

PANEL_PATH = "03_tabular/ward_year_panel.csv"


def load_panel(path=PANEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected ward-year panel at {path}. This file must be built "
            "from real GEE zonal-statistics exports and real tabular "
            "sources before any analysis below can run - see the "
            "Reproducibility Plan for the exact columns expected."
        )
    df = pd.read_csv(path)
    required = {"year", "ward_id", "ndvi_mean", "ndbi_mean", "lst_day_mean", "green_pct", "builtup_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------
# 1. DESCRIPTIVE / TREND TABLES  (Section 7, Section 19 "Descriptive")
# ---------------------------------------------------------------------
def yearly_summary(df):
    """Reproduces the Year / Green% / Built-up% table from Section 7."""
    return (
        df.groupby("year")
        .agg(
            green_pct_mean=("green_pct", "mean"),
            builtup_pct_mean=("builtup_pct", "mean"),
            ndvi_mean=("ndvi_mean", "mean"),
            lst_day_mean=("lst_day_mean", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# 2. CORRELATION / REGRESSION  (Section 9, Section 19)
# ---------------------------------------------------------------------
def correlation_matrix(df, cols):
    from scipy.stats import pearsonr, spearmanr

    results = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            sub = df[[c1, c2]].dropna()
            if len(sub) < 10:
                results.append({"var1": c1, "var2": c2, "n": len(sub), "note": "insufficient n - not computed"})
                continue
            r_p, p_p = pearsonr(sub[c1], sub[c2])
            r_s, p_s = spearmanr(sub[c1], sub[c2])
            results.append(
                {"var1": c1, "var2": c2, "n": len(sub), "pearson_r": r_p, "pearson_p": p_p, "spearman_r": r_s, "spearman_p": p_s}
            )
    return pd.DataFrame(results)


def lst_regression(df):
    """LST = b0 + b1*NDVI + b2*NDBI + b3*PopulationDensity + e (Section 19)."""
    import statsmodels.formula.api as smf

    needed = {"lst_day_mean", "ndvi_mean", "ndbi_mean", "population_density"}
    if not needed.issubset(df.columns) or df[list(needed)].dropna().shape[0] < 30:
        print("Skipping LST regression: required columns/sample size not met. "
              "Do not force a regression on an underpowered sample.")
        return None
    model = smf.ols("lst_day_mean ~ ndvi_mean + ndbi_mean + population_density", data=df).fit()
    print(model.summary())
    # Report residual diagnostics rather than assuming they're fine:
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print("Breusch-Pagan heteroscedasticity test p-value:", bp_test[1])
    return model


# ---------------------------------------------------------------------
# 3. ML PREDICTION FOR 2030 SCENARIO  (Sections 16-18)
# ---------------------------------------------------------------------
def train_and_validate_rf(df, target_col, feature_cols, test_year_holdout=2025):
    """
    Temporal hold-out validation: train on all years before
    `test_year_holdout`, test on that year, THEN retrain on the full
    2000-2025 series before producing the actual 2030 prediction.
    Never evaluate performance on the same data used to train.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train = df[df["year"] < test_year_holdout].dropna(subset=feature_cols + [target_col])
    test = df[df["year"] == test_year_holdout].dropna(subset=feature_cols + [target_col])

    if len(train) < 30 or len(test) < 5:
        print("Skipping RF training: insufficient rows for a meaningful temporal hold-out. "
              "With only ~6 historical time points x N wards, check whether N wards is "
              "large enough (e.g. >=30-50) before trusting this model.")
        return None, None

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(train[feature_cols], train[target_col])
    preds = model.predict(test[feature_cols])

    metrics = {
        "MAE": mean_absolute_error(test[target_col], preds),
        "RMSE": mean_squared_error(test[target_col], preds, squared=False),
        "R2": r2_score(test[target_col], preds),
    }
    print(f"Hold-out validation ({test_year_holdout}) for {target_col}:", metrics)
    return model, metrics


def predict_2030(df, target_col, feature_cols, feature_2025_2030_assumptions):
    """
    feature_2025_2030_assumptions: dict of {feature_name: projected_2030_value_per_ward}
    built from a simple linear trend extrapolation of each feature
    2000-2025 (or better, its own small model) - NOT fabricated by hand.
    This function only wraps prediction + reports an uncertainty range
    from the ensemble's individual tree predictions, per Section 18.
    """
    from sklearn.ensemble import RandomForestRegressor

    full_train = df.dropna(subset=feature_cols + [target_col])
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(full_train[feature_cols], full_train[target_col])

    X_2030 = pd.DataFrame(feature_2025_2030_assumptions)
    tree_preds = np.stack([t.predict(X_2030[feature_cols]) for t in model.estimators_])
    point_estimate = tree_preds.mean(axis=0)
    lower = np.percentile(tree_preds, 5, axis=0)
    upper = np.percentile(tree_preds, 95, axis=0)

    result = X_2030.copy()
    result[f"{target_col}_2030_pred"] = point_estimate
    result[f"{target_col}_2030_p05"] = lower
    result[f"{target_col}_2030_p95"] = upper
    return result


if __name__ == "__main__":
    # This will raise a clear FileNotFoundError until real data exist -
    # that is intentional, not a bug.
    panel = load_panel()
    print(yearly_summary(panel))
