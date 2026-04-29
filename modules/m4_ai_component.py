"""
Module M4 — Anomaly Detector (AI Component)

Identifies Bitcoin blocks whose inter-arrival times are statistically
abnormal under the assumption that mining follows a Poisson process.

Statistical Reasoning (C2 Criterion)
──────────────────────────────────────
Bitcoin mining is a memoryless Bernoulli process: each SHA256 hash
attempt independently succeeds with probability p ≈ target/2²⁵⁶.
In the large-N limit, waiting times between successes converge to an
exponential distribution with rate λ = 1/mean.

Under this model, the CDF gives the probability of seeing an
inter-arrival time ≤ t:

    F(t) = 1 − exp(−λ × t)

So p_value = 1 − F(t) = exp(−λ × t) is the probability of observing
a gap *at least as large* as t under the null model.

Anomaly rules:
  • Very long gap (p_value < 0.01): fewer than 1 % of blocks under
    the exponential model have this gap or larger.  May indicate a
    mining difficulty spike, network partition, or empty mempool.
  • Very fast block (t < 60 s): almost certainly two blocks found
    nearly simultaneously — common in large mining pools.

Limitations:
  • The exponential model assumes a constant, uniform hash rate.
    In reality hash rate varies (miners join/leave, hardware upgrades).
  • Short observation windows (200 blocks ≈ 33 hours) may not capture
    longer-term trends.
  • "Anomalous" clusters may indicate coordinated pool behavior rather
    than anything adversarial.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import BlockchainAPIError, get_last_n_blocks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BLOCKS = 200              # sample size for anomaly detection
FAST_BLOCK_THRESHOLD = 60   # seconds — suspiciously fast
SLOW_PVALUE_THRESHOLD = 0.01  # p-value below this = statistically rare long gap


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_inter_arrivals(blocks: list[dict]) -> pd.DataFrame:
    """
    Given a list of block dicts (most-recent first from the API),
    compute inter-arrival times and annotate each with its height.

    Returns a DataFrame with columns:
        height, timestamp, inter_arrival, p_value, is_anomaly, anomaly_reason
    """
    # Sort ascending by timestamp so differences are positive.
    sorted_blocks = sorted(blocks, key=lambda b: b["timestamp"])

    heights = [b["height"] for b in sorted_blocks]
    timestamps = [b["timestamp"] for b in sorted_blocks]

    inter_arrivals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    # Maximum Likelihood Estimate of λ for exponential distribution.
    # MLE: λ̂ = n / Σtᵢ = 1 / mean(t)
    arr = np.array(inter_arrivals, dtype=float)
    lambda_hat = 1.0 / arr.mean()

    # Compute survival probability for each observation:
    # P(T ≥ t) = exp(−λ × t)  ← probability of a gap this large or larger
    p_values = np.exp(-lambda_hat * arr)

    # Anomaly flags
    is_anomaly = (p_values < SLOW_PVALUE_THRESHOLD) | (arr < FAST_BLOCK_THRESHOLD)
    reasons = []
    for t, pv in zip(arr, p_values):
        r = []
        if pv < SLOW_PVALUE_THRESHOLD:
            r.append(f"Rare long gap (p={pv:.4f})")
        if t < FAST_BLOCK_THRESHOLD:
            r.append(f"Very fast block ({t:.0f}s)")
        reasons.append(", ".join(r) if r else "Normal")

    df = pd.DataFrame(
        {
            # Each inter-arrival is associated with the *later* block.
            "height": heights[1:],
            "timestamp": pd.to_datetime(timestamps[1:], unit="s"),
            "inter_arrival": arr,
            "p_value": p_values,
            "is_anomaly": is_anomaly,
            "anomaly_reason": reasons,
        }
    )
    return df, lambda_hat


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the M4 Anomaly Detector tab."""

    st.header("🤖 M4 — Block Anomaly Detector (AI Component)")
    st.caption(
        "Statistical anomaly detection on inter-arrival times using an "
        "exponential null model fitted to real blockchain data."
    )

    # ------------------------------------------------------------------
    # Fetch data
    # ------------------------------------------------------------------
    try:
        blocks = get_last_n_blocks(N_BLOCKS)
        st.session_state["m4_blocks"] = blocks
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ API unavailable — showing cached data. ({exc})")
        blocks = st.session_state.get("m4_blocks")
        if blocks is None:
            st.error("No cached data available.")
            return

    if len(blocks) < 10:
        st.error("Not enough blocks to run anomaly detection.")
        return

    df, lambda_hat = compute_inter_arrivals(blocks)
    mean_arrival = 1.0 / lambda_hat

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------
    st.subheader("📐 Fitted Exponential Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blocks analysed", len(blocks))
    c2.metric("Fitted λ⁻¹ (mean gap)", f"{mean_arrival:.1f} s")
    c3.metric("Expected mean gap", "600 s")
    n_anom = int(df["is_anomaly"].sum())
    anomaly_rate = n_anom / len(df) * 100
    c4.metric("Anomalies detected", f"{n_anom} ({anomaly_rate:.1f}%)")

    expected_anomaly_rate = (
        SLOW_PVALUE_THRESHOLD * 100 + (math.exp(-lambda_hat * FAST_BLOCK_THRESHOLD)) * 100
    )
    st.caption(
        f"Expected anomaly rate under the null model ≈ "
        f"{expected_anomaly_rate:.1f}% "
        f"(p < {SLOW_PVALUE_THRESHOLD} or t < {FAST_BLOCK_THRESHOLD}s). "
        f"Observed: {anomaly_rate:.1f}%."
    )

    st.divider()

    # ------------------------------------------------------------------
    # Scatter plot: height vs inter-arrival time
    # ------------------------------------------------------------------
    st.subheader("🔴 Anomaly Scatter Plot")

    normal = df[~df["is_anomaly"]]
    anomalous = df[df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal["height"],
            y=normal["inter_arrival"],
            mode="markers",
            name="Normal",
            marker=dict(color="#3d9be9", size=5, opacity=0.6),
            hovertemplate=(
                "Height: %{x:,}<br>"
                "Gap: %{y:.0f}s<br>"
                "<extra>Normal</extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomalous["height"],
            y=anomalous["inter_arrival"],
            mode="markers",
            name="Anomalous",
            marker=dict(color="#f44336", size=9, symbol="x", opacity=0.9),
            hovertemplate=(
                "Height: %{x:,}<br>"
                "Gap: %{y:.0f}s<br>"
                "<extra>ANOMALY</extra>"
            ),
        )
    )
    fig.add_hline(
        y=600,
        line_dash="dash",
        line_color="#ffd700",
        annotation_text="Expected 600 s",
        annotation_position="top right",
    )
    fig.add_hline(
        y=FAST_BLOCK_THRESHOLD,
        line_dash="dot",
        line_color="#ff9800",
        annotation_text=f"Fast threshold ({FAST_BLOCK_THRESHOLD}s)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Inter-Arrival Time per Block (last {len(blocks)} blocks)",
        xaxis_title="Block height",
        yaxis_title="Inter-arrival time (seconds)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Anomaly table
    # ------------------------------------------------------------------
    if n_anom > 0:
        st.subheader("📋 Anomalous Blocks")
        display_df = anomalous[
            ["height", "timestamp", "inter_arrival", "p_value", "anomaly_reason"]
        ].copy()
        display_df = display_df.rename(
            columns={
                "height": "Block Height",
                "timestamp": "Timestamp (UTC)",
                "inter_arrival": "Gap (s)",
                "p_value": "p-value",
                "anomaly_reason": "Reason",
            }
        )
        display_df["Gap (s)"] = display_df["Gap (s)"].round(1)
        display_df["p-value"] = display_df["p-value"].map("{:.6f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No anomalous blocks detected in this sample window.")

    # ------------------------------------------------------------------
    # Temporal clustering analysis
    # ------------------------------------------------------------------
    st.subheader("🕐 Temporal Clustering of Anomalies")
    st.caption(
        "Do anomalous blocks cluster in time? Clustering may indicate "
        "mining pool behaviour, network events, or hash-rate changes."
    )

    if n_anom >= 2:
        anom_times = anomalous["timestamp"].sort_values()
        gaps_between_anomalies = (
            anom_times.diff().dropna().dt.total_seconds() / 3600
        )
        fig_cluster = go.Figure()
        fig_cluster.add_trace(
            go.Scatter(
                x=anomalous["timestamp"],
                y=anomalous["inter_arrival"],
                mode="markers+lines",
                marker=dict(color="#f44336", size=8),
                line=dict(color="rgba(244,67,54,0.3)", dash="dot"),
                name="Anomalous gap",
            )
        )
        fig_cluster.update_layout(
            title="Anomalous Block Gaps Over Time",
            xaxis_title="Date",
            yaxis_title="Gap (seconds)",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        mean_between = gaps_between_anomalies.mean()
        st.caption(
            f"Mean time between anomalous blocks: {mean_between:.1f} hours. "
            + (
                "⚠️ Anomalies are clustered — may indicate a correlated event."
                if mean_between < 2
                else "Anomalies appear spread out — likely independent events."
            )
        )
    else:
        st.info("Fewer than 2 anomalies — no clustering analysis possible.")

    # ------------------------------------------------------------------
    # Model explanation
    # ------------------------------------------------------------------
    with st.expander("🔬 Statistical methodology"):
        st.markdown(
            r"""
**Null model:** Inter-arrival times are i.i.d. Exponential(λ) where
λ = 1/mean(observed gaps).

**Anomaly criterion:**
- `p_value = exp(−λ × t)` = probability of a gap ≥ t under null model.
- Flag if `p_value < 0.01` (rare long gap) **or** `t < 60 s` (very fast).

**Evaluation metrics:**
| Metric | Value |
|---|---|
| Anomaly rate | $(n_{anomalies} / n_{total}) \times 100\%$ |
| Expected rate (null) | $\approx 1\% + P(T < 60s)$ |
| Model: | Exponential MLE |

**Limitations:**
1. Assumes constant hash rate — invalid during difficulty transitions.
2. Only 200 blocks ≈ 33 hours of data; longer windows reduce false positives.
3. The exponential model treats all miners as a single aggregate process;
   real mining pools introduce short-range correlations.
            """
        )
