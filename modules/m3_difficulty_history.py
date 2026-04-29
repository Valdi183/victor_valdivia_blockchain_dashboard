"""
Module M3 — Difficulty History

Displays historical Bitcoin mining difficulty over the last 2 years:

  1. Line chart with difficulty on a linear/log scale.
  2. Vertical markers at each 2016-block difficulty adjustment epoch.
  3. Bar chart showing the actual/expected block time ratio per period.

Background: Satoshi's Difficulty Adjustment Algorithm (DAA)
────────────────────────────────────────────────────────────
Every 2016 blocks (≈ 2 weeks at 10 min/block), Bitcoin recalculates
the target so that the average inter-block time stays close to 600 s:

    new_difficulty = old_difficulty × (expected_time / actual_time)
                   = old_difficulty × (2016 × 600 s / actual_elapsed)

The adjustment is clamped to [0.25×, 4×] the previous difficulty to
prevent sudden 99 % drops or runaway increases (an anti-manipulation
guard added by Satoshi — see Bitcoin whitepaper §3 and the original
Bitcoin Core source `src/pow.cpp`).

Reference: Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash
           System. https://bitcoin.org/bitcoin.pdf  Section 3, 4.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from api.blockchain_client import BlockchainAPIError, get_difficulty_history

# One difficulty period = 2016 blocks ≈ 2 weeks
BLOCKS_PER_PERIOD = 2016
EXPECTED_PERIOD_SECONDS = BLOCKS_PER_PERIOD * 600   # 1 209 600 s ≈ 14 days


def render() -> None:
    """Render the M3 Difficulty History tab."""

    st.header("📈 M3 — Difficulty History")
    st.caption(
        "Historical Bitcoin mining difficulty with adjustment-epoch markers "
        "and per-period block-time ratios (last 2 years)."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col_l, col_r = st.columns([1, 2])
    with col_l:
        log_scale = st.toggle("Log scale for difficulty", value=False, key="m3_log")
    with col_r:
        st.markdown(
            "_Each adjustment period = 2016 blocks ≈ 2 weeks. "
            "Vertical dashed lines mark each epoch boundary._"
        )

    # ------------------------------------------------------------------
    # Fetch data
    # ------------------------------------------------------------------
    try:
        df = get_difficulty_history()
        st.session_state["m3_df"] = df
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ API unavailable — showing cached data. ({exc})")
        df = st.session_state.get("m3_df")
        if df is None:
            st.error("No cached data available. Please try again later.")
            return

    if df is None or df.empty:
        st.error("No difficulty data returned.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Identify adjustment epochs
    # ------------------------------------------------------------------
    # The blockchain.info API returns one data point per adjustment period
    # (each ~2 weeks). We annotate each transition.
    df["period_index"] = range(len(df))
    df["pct_change"] = df["difficulty"].pct_change() * 100

    # ------------------------------------------------------------------
    # Section 1 — Difficulty line chart
    # ------------------------------------------------------------------
    st.subheader("🔷 Difficulty Over Time")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["difficulty"],
            mode="lines",
            name="Difficulty",
            line=dict(color="#3d9be9", width=2),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Difficulty: %{y:,.0f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add vertical markers at every adjustment (every data point is one epoch).
    for _, row in df.iterrows():
        fig.add_vline(
            x=row["timestamp"],
            line_dash="dot",
            line_color="rgba(255,255,255,0.08)",
            line_width=1,
        )

    # Highlight the 3 biggest increases and decreases.
    top_ups = df.nlargest(3, "pct_change")
    top_downs = df.nsmallest(3, "pct_change")
    for _, row in top_ups.iterrows():
        if not np.isnan(row["pct_change"]):
            fig.add_annotation(
                x=row["timestamp"],
                y=row["difficulty"],
                text=f"+{row['pct_change']:.1f}%",
                showarrow=True,
                arrowhead=2,
                font=dict(color="#4caf50", size=10),
                arrowcolor="#4caf50",
            )
    for _, row in top_downs.iterrows():
        if not np.isnan(row["pct_change"]):
            fig.add_annotation(
                x=row["timestamp"],
                y=row["difficulty"],
                text=f"{row['pct_change']:.1f}%",
                showarrow=True,
                arrowhead=2,
                font=dict(color="#f44336", size=10),
                arrowcolor="#f44336",
            )

    fig.update_layout(
        title="Bitcoin Mining Difficulty (2-Year History)",
        xaxis_title="Date",
        yaxis_title="Difficulty" + (" (log scale)" if log_scale else ""),
        yaxis_type="log" if log_scale else "linear",
        template="plotly_dark",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 2 — Per-period block time ratio bar chart
    # ------------------------------------------------------------------
    st.subheader("⚖️ Block Time Ratio per Adjustment Period")
    st.caption(
        "Ratio = actual avg block time / 600 s. "
        "Bars > 1 (red): blocks were slower → difficulty dropped next period. "
        "Bars < 1 (green): blocks were faster → difficulty rose next period."
    )

    # The blockchain.info API gives difficulty at each epoch; we infer the
    # time ratio from the difficulty change itself using the inverse of the
    # DAA formula: ratio = old_difficulty / new_difficulty
    # (clamped reverse: if diff increased by factor k, ratio = 1/k)
    df["ratio"] = df["difficulty"].shift(1) / df["difficulty"]

    # Drop first row (NaN) and clamp to [0.25, 4] per Bitcoin's DAA rules.
    ratio_df = df.dropna(subset=["ratio"]).copy()
    ratio_df["ratio"] = ratio_df["ratio"].clip(0.25, 4.0)
    ratio_df["color"] = ratio_df["ratio"].apply(
        lambda r: "#f44336" if r > 1.0 else "#4caf50"
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=ratio_df["timestamp"],
            y=ratio_df["ratio"],
            marker_color=ratio_df["color"],
            name="Time ratio",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Ratio: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig2.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#ffd700",
        annotation_text="Expected (ratio = 1)",
        annotation_position="top right",
    )
    fig2.update_layout(
        title="Actual/Expected Block Time Ratio per Period",
        xaxis_title="Period end date",
        yaxis_title="Ratio (actual / 600 s)",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Adjustment Statistics")

    n_periods = len(ratio_df)
    n_increase = (ratio_df["ratio"] < 1.0).sum()
    n_decrease = (ratio_df["ratio"] > 1.0).sum()
    max_increase = df["pct_change"].max()
    max_decrease = df["pct_change"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Periods shown", n_periods)
    c2.metric("Difficulty increases", int(n_increase))
    c3.metric("Difficulty decreases", int(n_decrease))
    c4.metric("Largest single increase", f"+{max_increase:.1f}%")

    with st.expander("📖 Satoshi's Difficulty Adjustment Formula"):
        st.markdown(
            r"""
Every 2016 blocks Bitcoin recalculates the mining target:

$$\text{new\_difficulty} = \text{old\_difficulty} \times \frac{2016 \times 600\,\text{s}}{\text{actual\_elapsed}}$$

Equivalently, in terms of the target (smaller = harder):

$$\text{new\_target} = \text{old\_target} \times \frac{\text{actual\_elapsed}}{2016 \times 600\,\text{s}}$$

**Clamping rule (anti-manipulation):**

$$\text{new\_difficulty} \in [0.25 \times \text{old}, \; 4 \times \text{old}]$$

This prevents a sudden hash-rate spike or crash from making the chain
unmineable or trivially easy within a single adjustment cycle.

*Reference: Nakamoto (2008), §3; Bitcoin Core `src/pow.cpp`.*
            """
        )
