"""
Module M6 — Security Score: Cost of a 51% Attack

Estimates the real-time cost (USD/hour) required to execute a 51% attack on
Bitcoin and visualises how confirmation depth reduces the attacker's success
probability using Nakamoto's (2008) §11 formula.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What is a 51% attack?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An attacker controlling > 50% of the global hash rate can, in expectation,
produce blocks faster than the honest network. This allows double-spending:
broadcast a payment, wait for a merchant to confirm it, then mine a secret
longer chain that omits that payment.

The cost model here estimates what it would take to rent enough ASIC hardware
to match 51% of the current hash rate. In practice, no such rental market
exists at this scale, so the estimate represents a lower bound.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nakamoto (2008) §11 — Attack success probability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Let:
  p = honest hash fraction,  q = attacker fraction (p + q = 1)
  z = number of confirmations the merchant waits for

While the honest chain gains z blocks, the attacker mines in secret.
The number of attacker blocks k follows Poisson(λ), λ = z·(q/p).

The probability the attacker can ever catch up (Gambler's ruin result):

    P_success(z) = 1 − Σ_{k=0}^{z} [ Poisson(k; λ) × (1 − (q/p)^{z−k}) ]

For q < 0.5, P_success(z) → 0 exponentially as z increases.
For q ≥ 0.5, P_success = 1 for all z.

Reference: Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash
           System. Section 11. https://bitcoin.org/bitcoin.pdf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hardware reference — Antminer S19 XP (Bitmain, 2022)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hash rate : 140 TH/s  (1 TH/s = 10¹² hashes/second)
  Power draw: 3 010 W
  Unit cost : ~$2 000 (secondary market, 2024)
These values are conservative estimates from public ASIC market data.
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

from api.blockchain_client import BlockchainAPIError, get_latest_block
from modules.m1_proof_of_work import decode_bits, target_to_difficulty

# ---------------------------------------------------------------------------
# Hardware constants (Antminer S19 XP — justification in module docstring)
# ---------------------------------------------------------------------------
ASIC_HASH_RATE_THS   = 140          # TH/s per unit
ASIC_POWER_WATTS     = 3_010        # watts per unit
ASIC_PRICE_USD       = 2_000        # USD per unit (secondary market 2024)
ELECTRICITY_USD_KWH  = 0.05         # USD/kWh — industrial mining facility rate

DIFFICULTY_1_TARGET = 0x00000000FFFF0000_0000000000000000_0000000000000000_0000000000000000


# ---------------------------------------------------------------------------
# Cryptographic / economic helpers
# ---------------------------------------------------------------------------

def nakamoto_attack_probability(q: float, z: int) -> float:
    """
    Compute P_success(q, z): probability an attacker with hash fraction q
    can rewrite a chain that already has z confirmations.

    Uses the exact Nakamoto (2008) §11 formula with the Poisson distribution.
    scipy.stats.poisson.pmf handles numerical stability for large z and λ.

    Parameters
    ----------
    q : attacker's fraction of total hash rate  (0 < q < 1)
    z : confirmation depth (number of blocks the honest chain is ahead)
    """
    p = 1.0 - q
    if q >= p:
        return 1.0          # attacker dominates — always wins eventually
    if z == 0:
        return 1.0          # zero confirmations → trivially revertible

    # λ = expected attacker blocks while honest chain mines z blocks.
    lam = z * (q / p)

    # Σ_{k=0}^{z} Poisson(k; λ) × (1 − (q/p)^{z−k})
    # represents the probability the attacker FAILS: they mine k blocks
    # but are still (z-k) behind, and the geometric catch-up probability
    # for a (z-k) deficit is (q/p)^(z-k).
    total = sum(
        float(poisson.pmf(k, lam)) * (1.0 - (q / p) ** (z - k))
        for k in range(z + 1)
    )
    return max(0.0, 1.0 - total)


def estimate_attack_cost(
    network_hash_rate_ths: float,
    electricity_usd_kwh: float = ELECTRICITY_USD_KWH,
) -> dict:
    """
    Estimate the cost in USD/hour to supply 51% of the network hash rate.

    Two cost components:
    1. Electricity: ongoing operational cost (dominant for large attacks).
    2. Hardware:   amortised capital cost (assumes 3-year hardware lifetime).

    Parameters
    ----------
    network_hash_rate_ths : total network hash rate in TH/s
    electricity_usd_kwh   : electricity price in USD per kWh
    """
    attacker_ths = network_hash_rate_ths * 0.51   # need to exceed 50%
    n_asics = math.ceil(attacker_ths / ASIC_HASH_RATE_THS)

    # Electricity cost per hour
    total_kw = n_asics * ASIC_POWER_WATTS / 1_000
    electricity_per_hour = total_kw * electricity_usd_kwh

    # Hardware cost amortised over 3 years = 26 280 hours
    hardware_total = n_asics * ASIC_PRICE_USD
    hardware_per_hour = hardware_total / (3 * 365 * 24)

    total_per_hour = electricity_per_hour + hardware_per_hour

    return {
        "n_asics": n_asics,
        "attacker_ths": attacker_ths,
        "total_kw": total_kw,
        "electricity_per_hour_usd": electricity_per_hour,
        "hardware_total_usd": hardware_total,
        "hardware_per_hour_usd": hardware_per_hour,
        "total_per_hour_usd": total_per_hour,
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the M6 Security Score tab."""

    st.header("🛡️ M6 — Security Score: Cost of a 51% Attack")
    st.caption(
        "Real-time estimate of what it costs to attack Bitcoin, "
        "and how confirmation depth neutralises the attacker."
    )

    # ------------------------------------------------------------------
    # Fetch latest block for hash rate
    # ------------------------------------------------------------------
    try:
        latest = get_latest_block()
        st.session_state["m6_latest"] = latest
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ API unavailable — showing cached data. ({exc})")
        latest = st.session_state.get("m6_latest")
        if latest is None:
            st.error("No cached data available.")
            return

    bits_val = latest.get("bits")
    if bits_val is None:
        st.error("Missing `bits` field in block data.")
        return

    target = decode_bits(bits_val)
    difficulty = target_to_difficulty(target)

    # Hash rate: difficulty × 2³² / 600 s (same formula as M1).
    hash_rate_hps = difficulty * (2 ** 32) / 600
    hash_rate_ths = hash_rate_hps / 1e12     # TH/s
    hash_rate_ehs = hash_rate_hps / 1e18     # EH/s

    # ------------------------------------------------------------------
    # Section 1 — Current network stats
    # ------------------------------------------------------------------
    st.subheader("📡 Current Network Hash Rate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Block Height",    f"{latest['height']:,}")
    c2.metric("Network Hash Rate", f"{hash_rate_ehs:.2f} EH/s",
              help="Estimated from difficulty × 2³² / 600 s")
    c3.metric("Mining Difficulty", f"{difficulty:,.0f}")

    st.divider()

    # ------------------------------------------------------------------
    # Section 2 — Attack cost breakdown
    # ------------------------------------------------------------------
    st.subheader("💰 Estimated Cost to Execute a 51% Attack")

    col_e, col_h = st.columns([1, 2])
    with col_h:
        electricity_price = st.slider(
            "Electricity price (USD/kWh)",
            min_value=0.01, max_value=0.20,
            value=ELECTRICITY_USD_KWH, step=0.01,
            key="m6_elec_price",
        )

    cost = estimate_attack_cost(hash_rate_ths, electricity_price)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Electricity cost / hour",
        f"${cost['electricity_per_hour_usd']:,.0f}",
        help=f"{cost['n_asics']:,} ASICs × {ASIC_POWER_WATTS}W @ ${electricity_price}/kWh",
    )
    c2.metric(
        "Hardware cost / hour",
        f"${cost['hardware_per_hour_usd']:,.0f}",
        help=f"${cost['hardware_total_usd']:,.0f} capex amortised over 3 years",
    )
    c3.metric(
        "Total cost / hour",
        f"${cost['total_per_hour_usd']:,.0f}",
    )

    with st.expander("📋 Hardware assumption details"):
        st.markdown(
            f"""
**Reference ASIC: Antminer S19 XP** (Bitmain, 2022 — current generation)

| Parameter | Value | Source |
|---|---|---|
| Hash rate | {ASIC_HASH_RATE_THS} TH/s | Bitmain datasheet |
| Power draw | {ASIC_POWER_WATTS:,} W | Bitmain datasheet |
| Unit price | ${ASIC_PRICE_USD:,} | Secondary market, 2024 |
| Electricity | ${electricity_price}/kWh | Industrial mining rate (adjustable above) |

**Units needed to reach 51%:**
- Attacker target: {cost['attacker_ths']/1e6:.2f} EH/s
  = {cost['attacker_ths']:,.0f} TH/s
- ASICs required: **{cost['n_asics']:,}** units
- Total power draw: {cost['total_kw']:,.0f} kW = {cost['total_kw']/1_000:.1f} MW

Note: this is a lower-bound estimate. Real attacks also require
infrastructure, cooling, and assume no hardware is pre-owned.
            """
        )

    st.divider()

    # ------------------------------------------------------------------
    # Section 3 — Nakamoto §11 attack probability curves
    # ------------------------------------------------------------------
    st.subheader("📉 Attack Success Probability vs Confirmation Depth")
    st.caption(
        "Based on Nakamoto (2008) §11. "
        "Each curve shows how quickly deeper confirmations make the attack impractical."
    )

    z_values = list(range(0, 21))   # confirmation depths 0..20
    q_values = [0.10, 0.20, 0.30]   # three attacker scenarios
    colors    = ["#4caf50", "#ff9800", "#f44336"]  # green, orange, red

    fig = go.Figure()

    for q, color in zip(q_values, colors):
        probs = [nakamoto_attack_probability(q, z) for z in z_values]
        fig.add_trace(go.Scatter(
            x=z_values,
            y=[p * 100 for p in probs],
            mode="lines+markers",
            name=f"q = {q:.0%} hash rate",
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate=(
                f"q = {q:.0%}<br>"
                "z = %{x} confirmations<br>"
                "P(success) = %{y:.4f}%<extra></extra>"
            ),
        ))

    # Mark the conventional "6 confirmations" threshold.
    fig.add_vline(
        x=6, line_dash="dot", line_color="#ffd700",
        annotation_text="6 confirmations (convention)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.1, line_dash="dot", line_color="rgba(255,255,255,0.3)",
        annotation_text="0.1% threshold",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Attack Success Probability — Nakamoto (2008) §11",
        xaxis_title="Confirmation depth z (blocks)",
        yaxis_title="P(attack succeeds) [%]",
        yaxis_type="log",
        legend=dict(x=0.6, y=0.9),
        template="plotly_dark",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table of P values at z = 1, 3, 6, 10
    st.markdown("**P(attack succeeds) at key confirmation depths:**")
    key_depths = [1, 3, 6, 10, 20]
    table_rows = []
    for q in q_values:
        row = {"q (attacker share)": f"{q:.0%}"}
        for z in key_depths:
            p = nakamoto_attack_probability(q, z)
            row[f"z={z}"] = f"{p*100:.4f}%"
        table_rows.append(row)
    import pandas as pd
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with st.expander("📖 Nakamoto §11 formula derivation"):
        st.markdown(
            r"""
**Setup:** honest chain mines at rate $p = 1 - q$; attacker at rate $q$.
The attacker starts mining secretly while the honest chain gains $z$ blocks.

**Key insight (Gambler's ruin):** if the attacker is $k$ blocks behind,
the probability of ever catching up is $(q/p)^k$ for $q < p$.

**Poisson model:** while the honest chain mines $z$ blocks,
the attacker mines $k \sim \text{Poisson}(\lambda)$ blocks,
where $\lambda = z \cdot (q/p)$ (expected attacker output).

**Nakamoto formula:**
$$P_{\text{success}}(z) = 1 - \sum_{k=0}^{z}
\underbrace{\frac{e^{-\lambda}\lambda^k}{k!}}_{\text{Poisson PMF}}
\cdot \left(1 - \left(\frac{q}{p}\right)^{z-k}\right)$$

For $q < 0.5$, $P_{\text{success}}(z) \to 0$ exponentially in $z$.
At $q = 0.10$, six confirmations reduce the success probability below 0.1%.

*Reference: Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
Section 11.*
            """
        )
