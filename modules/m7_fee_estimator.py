"""
Module M7 — Fee Estimator (AI Component #2)

Trains a supervised Gradient Boosting regression model to predict the next
period's median Bitcoin transaction fee (sat/vByte) from time-based and
lag features derived from real blockchain data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why this model complements M4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
M4 (Anomaly Detector):
  • Data source  : block inter-arrival times (timing of blocks)
  • Task         : unsupervised anomaly detection
  • Model        : statistical threshold (exponential distribution)
  • Metrics      : KS test, anomaly rate

M7 (Fee Estimator):
  • Data source  : historical transaction fee rates (economic activity)
  • Task         : supervised 1-step-ahead regression
  • Model        : Gradient Boosting Regressor (tree ensemble)
  • Metrics      : MAE, RMSE, R²

Together they monitor both the *network layer* (block timing) and the
*economic layer* (fee market) of Bitcoin.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why Gradient Boosting?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bitcoin fees are driven by non-linear interactions:
  • Time-of-day effects (peak usage hours)
  • Day-of-week effects (weekday vs weekend)
  • Autocorrelation (fees cluster in congestion episodes)

GBR handles these without linearity assumptions. It is more interpretable
than an LSTM and requires no hyperparameter tuning for a proof-of-concept.
Linear regression was rejected because fee rate distributions are right-skewed
and exhibit interactions between lag features and time-of-day.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source  : mempool.space /v1/mining/blocks/fee-rates/1w
Target  : avgFee_50 of the NEXT period (1-step-ahead prediction, sat/vByte)
Features: hour_of_day, day_of_week, lag_1, lag_3, lag_6, rolling_std_6
Split   : first 80% = train, last 20% = test (temporal — no shuffling to
          prevent data leakage from future periods into training).
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from api.blockchain_client import (
    BlockchainAPIError,
    get_fee_rates_history,
    get_recommended_fees,
    get_last_n_blocks,
)
from modules.m4_ai_component import compute_inter_arrivals

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURES = ["hour", "day_of_week", "lag_1", "lag_3", "lag_6", "rolling_std_6"]
TARGET    = "target_fee"

# GBR hyperparameters — chosen for bias-variance balance on ~800 samples.
GBR_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,       # stochastic GBR reduces overfitting
    random_state=42,
)


def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer lag and time features from the raw fee rate DataFrame.

    Features
    --------
    hour          : hour of day (0–23) — captures intraday fee cycles
    day_of_week   : Mon=0 … Sun=6     — captures weekend vs weekday demand
    lag_1         : previous period's median fee (strongest autocorrelation)
    lag_3         : 3 periods ago (short-term trend)
    lag_6         : 6 periods ago (~1 hour for per-block data)
    rolling_std_6 : 6-period rolling std deviation (fee volatility signal)
    target_fee    : NEXT period's median fee (what we predict)

    All lag-based rows with NaN are dropped; the resulting split is temporal.
    """
    # Find the median fee column (mempool.space uses 'avgFee_50').
    fee_col = None
    for candidate in ("avgFee_50", "avg_fee_50"):
        if candidate in df_raw.columns:
            fee_col = candidate
            break
    if fee_col is None:
        # Fall back to any column containing "50"
        candidates = [c for c in df_raw.columns if "50" in c]
        if not candidates:
            raise ValueError(f"Cannot find median fee column in: {list(df_raw.columns)}")
        fee_col = candidates[0]

    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)
    df["fee"] = df[fee_col].astype(float)

    # Time features derived from the period timestamp.
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Lag features (past values as predictors).
    df["lag_1"] = df["fee"].shift(1)
    df["lag_3"] = df["fee"].shift(3)
    df["lag_6"] = df["fee"].shift(6)

    # Volatility feature: rolling standard deviation over last 6 periods.
    df["rolling_std_6"] = df["fee"].rolling(6, min_periods=2).std().fillna(0)

    # Target: next period's fee (1-step-ahead prediction).
    df[TARGET] = df["fee"].shift(-1)

    # Drop rows with NaN (due to lags at the start and target at the end).
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a GradientBoostingRegressor on the first 80% of samples and
    evaluate on the last 20% (temporal split — no shuffling).

    Returns a dict with: model, metrics (MAE, RMSE, R²), test arrays,
    feature importances, and split index.
    """
    n = len(df)
    if n < 30:
        raise ValueError(f"Insufficient data: {n} samples after feature engineering.")

    split = int(n * 0.80)
    X_train = df[FEATURES].iloc[:split].values
    y_train = df[TARGET].iloc[:split].values
    X_test  = df[FEATURES].iloc[split:].values
    y_test  = df[TARGET].iloc[split:].values

    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        "model":       model,
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "timestamps":  df["timestamp"].iloc[split:].values,
        "importances": dict(zip(FEATURES, model.feature_importances_)),
        "n_train":     split,
        "n_test":      n - split,
        "n_total":     n,
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the M7 Fee Estimator tab."""

    st.header("💸 M7 — Transaction Fee Estimator (AI Component #2)")
    st.caption(
        "Supervised Gradient Boosting regression model trained on real "
        "mempool fee data. Compared side-by-side with M4 (Anomaly Detector)."
    )

    # ------------------------------------------------------------------
    # Current recommended fees (live, on-demand)
    # ------------------------------------------------------------------
    st.subheader("📡 Current Recommended Fees")
    try:
        rec = get_recommended_fees()
        st.session_state["m7_rec_fees"] = rec
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ Could not fetch recommended fees: {exc}")
        rec = st.session_state.get("m7_rec_fees", {})

    if rec:
        cf1, cf2, cf3, cf4 = st.columns(4)
        cf1.metric("Fastest",    f"{rec.get('fastestFee', '?')} sat/vB")
        cf2.metric("30 min",     f"{rec.get('halfHourFee', '?')} sat/vB")
        cf3.metric("1 hour",     f"{rec.get('hourFee', '?')} sat/vB")
        cf4.metric("Economy",    f"{rec.get('economyFee', '?')} sat/vB")

    st.divider()

    # ------------------------------------------------------------------
    # Train / load model
    # ------------------------------------------------------------------
    st.subheader("🤖 Fee Prediction Model")

    train_btn = st.button("▶ Train / retrain model", key="m7_train_btn")
    has_model = "m7_result" in st.session_state

    if train_btn or not has_model:
        with st.spinner("Fetching fee rate history and training model…"):
            try:
                df_raw = get_fee_rates_history("1w")
            except BlockchainAPIError as exc:
                st.error(f"Cannot fetch fee history: {exc}")
                return

            try:
                df_feat = prepare_features(df_raw)
            except ValueError as exc:
                st.error(f"Feature engineering failed: {exc}")
                return

            try:
                result = train_model(df_feat)
                result["df"] = df_feat
            except ValueError as exc:
                st.error(str(exc))
                return

        st.session_state["m7_result"] = result
        st.success(
            f"✅ Model trained on {result['n_train']} samples, "
            f"evaluated on {result['n_test']} samples."
        )
    else:
        result = st.session_state["m7_result"]

    # ------------------------------------------------------------------
    # Section 1 — Model performance metrics
    # ------------------------------------------------------------------
    st.subheader("📐 Model Evaluation Metrics")

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("MAE",  f"{result['mae']:.2f} sat/vB",
               help="Mean Absolute Error — average prediction error in sat/vByte")
    cm2.metric("RMSE", f"{result['rmse']:.2f} sat/vB",
               help="Root Mean Squared Error — penalises large errors more than MAE")
    cm3.metric("R²",   f"{result['r2']:.3f}",
               help="Coefficient of determination; 1.0 = perfect, 0 = mean baseline")
    cm4.metric("Test samples", result["n_test"],
               help=f"Last 20% of {result['n_total']} total samples (temporal split)")

    # ------------------------------------------------------------------
    # Section 2 — Predicted vs actual chart
    # ------------------------------------------------------------------
    st.subheader("📊 Predicted vs Actual Fee Rate (Test Set)")

    ts = pd.to_datetime(result["timestamps"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=result["y_test"],
        mode="lines", name="Actual",
        line=dict(color="#3d9be9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=result["y_pred"],
        mode="lines", name="Predicted",
        line=dict(color="#f0383c", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Fee Rate Prediction vs Actual (sat/vByte) — Test Set",
        xaxis_title="Date",
        yaxis_title="Median fee rate (sat/vByte)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals histogram
    residuals = result["y_test"] - result["y_pred"]
    fig_res = go.Figure(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color="#3d9be9",
        opacity=0.75,
        name="Residuals",
    ))
    fig_res.add_vline(x=0, line_dash="dash", line_color="#ffd700",
                      annotation_text="Zero error")
    fig_res.update_layout(
        title="Residuals Distribution (actual − predicted)",
        xaxis_title="Error (sat/vByte)",
        yaxis_title="Count",
        template="plotly_dark",
        height=280,
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 3 — Feature importances
    # ------------------------------------------------------------------
    st.subheader("🔍 Feature Importances")

    imp = result["importances"]
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    fig_imp = go.Figure(go.Bar(
        x=[v for _, v in imp_sorted],
        y=[k for k, _ in imp_sorted],
        orientation="h",
        marker_color="#4caf50",
    ))
    fig_imp.update_layout(
        title="GBR Feature Importances",
        xaxis_title="Importance (mean decrease in impurity)",
        template="plotly_dark",
        height=280,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Section 4 — M4 vs M7 comparison panel
    # ------------------------------------------------------------------
    st.subheader("🔄 M4 vs M7 — AI Component Comparison")
    st.caption(
        "Both models use real Bitcoin data but address different layers of the network."
    )

    col_m4, col_m7 = st.columns(2)

    # --- M4 column ---
    with col_m4:
        st.markdown("### M4 — Anomaly Detector")
        st.markdown("**Type:** Unsupervised statistical model")
        st.markdown("**Data:** Block inter-arrival times (last 200 blocks)")
        st.markdown("**Algorithm:** Exponential distribution MLE + survival function")

        try:
            blocks_m4 = get_last_n_blocks(200)
            df_m4, lam = compute_inter_arrivals(blocks_m4)
            arr_m4 = df_m4["inter_arrival"].values
            ks_stat, ks_pval = stats.kstest(
                arr_m4, lambda x: 1.0 - np.exp(-lam * x)
            )
            n_anom = int(df_m4["is_anomaly"].sum())
            anom_rate = n_anom / len(df_m4) * 100
            st.metric("KS statistic",   f"{ks_stat:.4f}")
            st.metric("KS p-value",     f"{ks_pval:.4f}",
                      help="p > 0.05 → data consistent with exponential model")
            st.metric("Anomaly rate",   f"{anom_rate:.1f}%  ({n_anom}/{len(df_m4)})")
            st.metric("Fitted mean gap", f"{1/lam:.1f} s  (ideal: 600 s)")
        except Exception as exc:
            st.info(f"Could not load M4 data: {exc}")

    # --- M7 column ---
    with col_m7:
        st.markdown("### M7 — Fee Estimator")
        st.markdown("**Type:** Supervised regression")
        st.markdown("**Data:** Historical per-period fee rates (last 1 week)")
        st.markdown("**Algorithm:** Gradient Boosting Regressor (sklearn)")

        st.metric("MAE",          f"{result['mae']:.2f} sat/vB")
        st.metric("RMSE",         f"{result['rmse']:.2f} sat/vB")
        st.metric("R²",           f"{result['r2']:.3f}")
        st.metric("Training set", f"{result['n_train']} samples")

    st.divider()

    # Comparison summary table
    st.markdown("**Head-to-head comparison:**")
    comparison = pd.DataFrame([
        {"Dimension": "Problem type",    "M4 Anomaly Detector": "Detection",         "M7 Fee Estimator": "Regression"},
        {"Dimension": "Supervision",     "M4 Anomaly Detector": "Unsupervised",       "M7 Fee Estimator": "Supervised"},
        {"Dimension": "Data source",     "M4 Anomaly Detector": "Block timestamps",   "M7 Fee Estimator": "Fee rate history"},
        {"Dimension": "Key metric",      "M4 Anomaly Detector": "KS p-value",         "M7 Fee Estimator": "MAE / R²"},
        {"Dimension": "Interpretability","M4 Anomaly Detector": "High (parametric)",  "M7 Fee Estimator": "Medium (tree ensemble)"},
        {"Dimension": "Ground truth",    "M4 Anomaly Detector": "None (unsupervised)","M7 Fee Estimator": "Next-period fee"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    with st.expander("📖 Model justification and limitations"):
        st.markdown(
            f"""
**Why Gradient Boosting for fee prediction?**

1. **Non-linearity**: fee rates are right-skewed and driven by non-linear
   interactions (high-fee blocks cluster during specific hours AND during
   congestion episodes). GBR captures these without transformation.

2. **Feature interactions**: the model learns that `lag_1` matters more
   during high-`rolling_std_6` regimes (volatile periods), something linear
   regression cannot capture directly.

3. **No tuning needed**: default GBR hyperparameters with moderate depth
   generalise well on ~800-sample time series.

**Limitations:**
- The model does not capture sudden mempool explosions (e.g., Ordinals inscriptions)
  because they appear as outliers in the training set.
- 1-week training window is short; a longer history would improve R².
- Predicting fees requires mempool state (pending tx count, block space demand)
  which is not included here due to API constraints — adding it would improve MAE.

**Evaluation split:** chronological 80/20 — the test set is always the
most RECENT {result['n_test']} samples, ensuring no look-ahead bias.
            """
        )
