"""
Module M1 — Proof of Work Monitor

Displays live Bitcoin mining statistics derived from the last N blocks:

  1. Current difficulty (decoded manually from the `bits` compact field).
  2. Inter-arrival time histogram with theoretical exponential PDF overlay.
  3. Estimated network hash rate.

Cryptographic background (C1 criterion)
----------------------------------------
Bitcoin Proof of Work requires miners to find a nonce such that:

    SHA256(SHA256(header)) < target

The *target* is a 256-bit integer stored in compact form as `bits`:
  - byte 0  : exponent   (e)
  - bytes 1-3: coefficient (c, 3 bytes, big-endian)
  - target = c × 2^(8 × (e − 3))

Difficulty is defined relative to the genesis block target
(difficulty_1_target), which was chosen to require ~10 minutes per block
on the hardware available in 2009:

    difficulty_1_target = 0x00000000FFFF0000...0000  (26 trailing zeros)
    difficulty = difficulty_1_target / current_target

WHY are inter-arrival times exponentially distributed?
Mining is a memoryless Bernoulli process: each hash attempt succeeds with
independent probability p ≈ target/2^256.  Because each attempt is
independent (the Markov property / "memoryless" property), the number of
attempts until success follows a geometric distribution.  In the
continuous-time limit (attempts per second → ∞), this converges to an
exponential distribution with rate λ = 1/E[T] = 1/600 seconds.
This is analogous to radioactive decay or Poisson arrival processes.
"""

import hashlib
import math
import struct

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import expon

from api.blockchain_client import BlockchainAPIError, get_last_n_blocks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The genesis-block target (difficulty 1) used in all difficulty calculations.
# From Satoshi's original code: 0x00000000FFFF0000...0000 (big-endian).
DIFFICULTY_1_TARGET = 0x00000000FFFF0000_0000000000000000_0000000000000000_00000000_00000000

EXPECTED_BLOCK_TIME = 600  # seconds (10 minutes)


# ---------------------------------------------------------------------------
# Cryptographic helpers
# ---------------------------------------------------------------------------

def decode_bits(bits_int: int) -> int:
    """
    Decode a compact `bits` integer into the full 256-bit target integer.

    The compact format stores the target as:
        bits = (exponent << 24) | coefficient
    where
        exponent    = bits >> 24          (1 byte)
        coefficient = bits & 0x00FFFFFF   (3 bytes, big-endian significand)

    Reconstruction:
        target = coefficient × 2^(8 × (exponent − 3))

    This is equivalent to treating `coefficient` as the top 3 significant
    bytes of the target and shifting them into position.
    """
    exponent = (bits_int >> 24) & 0xFF
    coefficient = bits_int & 0x00FFFFFF
    target = coefficient * (2 ** (8 * (exponent - 3)))
    return target


def target_to_difficulty(target: int) -> float:
    """
    Convert a 256-bit target integer to the human-readable difficulty value.

    difficulty = difficulty_1_target / current_target

    A higher difficulty means a smaller target (harder to find a valid hash).
    """
    if target == 0:
        return float("inf")
    return DIFFICULTY_1_TARGET / target


def target_to_hex(target: int) -> str:
    """
    Format the 256-bit target as a 64-character zero-padded hex string.

    This makes the leading zeros — which embody the PoW requirement —
    visually obvious.  Every leading zero represents 4 bits of required work.
    """
    return f"{target:064x}"


def count_leading_zero_bits(hash_bytes: bytes) -> int:
    """Count how many leading *bits* (not nibbles) are zero in a hash."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            # bin() gives e.g. '0b10110100'; strip prefix and count leading 0s.
            count += 8 - len(bin(byte)) + 2  # +2 for '0b' prefix
            break
    return count


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(n_blocks: int = 50) -> None:
    """Render the M1 Proof of Work Monitor tab."""

    st.header("⛏️ M1 — Proof of Work Monitor")
    st.caption(
        "Live Bitcoin mining statistics derived from the last "
        f"{n_blocks} blocks."
    )

    # ------------------------------------------------------------------
    # Fetch data (with session_state fallback for API errors)
    # ------------------------------------------------------------------
    cache_key = f"m1_blocks_{n_blocks}"

    try:
        blocks = get_last_n_blocks(n_blocks)
        st.session_state[cache_key] = blocks
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ API unavailable — showing cached data. ({exc})")
        blocks = st.session_state.get(cache_key)
        if blocks is None:
            st.error("No cached data available. Please try again later.")
            return

    if not blocks:
        st.error("No block data returned.")
        return

    latest = blocks[0]

    # ------------------------------------------------------------------
    # Section 1 — Current difficulty and target
    # ------------------------------------------------------------------
    st.subheader("🔢 Current Difficulty & Target")

    bits_val = latest.get("bits")
    if bits_val is None:
        st.error("Block data missing `bits` field.")
        return

    target = decode_bits(bits_val)
    difficulty = target_to_difficulty(target)
    target_hex = target_to_hex(target)

    # Split target_hex into leading zeros and the rest for visual emphasis.
    leading_zeros = len(target_hex) - len(target_hex.lstrip("0"))
    zero_part = target_hex[:leading_zeros]
    sig_part = target_hex[leading_zeros:]

    col1, col2, col3 = st.columns(3)
    col1.metric("Block Height", f"{latest['height']:,}")
    col2.metric("Difficulty", f"{difficulty:,.2f}")
    col3.metric("Bits (compact)", hex(bits_val))

    st.markdown("**256-bit target (leading zeros highlighted):**")
    st.markdown(
        f'<code style="font-size:0.85em">'
        f'<span style="color:#f0383c;font-weight:bold">{zero_part}</span>'
        f'<span style="color:#e8e8e8">{sig_part}</span>'
        f"</code>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Leading zeros: **{leading_zeros} hex chars = {leading_zeros * 4} bits**. "
        "A valid block hash must be numerically below this target."
    )

    with st.expander("📖 How `bits` decodes to a target"):
        e = (bits_val >> 24) & 0xFF
        c = bits_val & 0x00FFFFFF
        st.markdown(
            f"""
The compact `bits` value `{hex(bits_val)}` is decoded as:

| Field | Value |
|---|---|
| Exponent (byte 0) | `{e}` |
| Coefficient (bytes 1-3) | `{hex(c)}` |
| Formula | `target = {hex(c)} × 2^(8 × ({e} − 3))` |
| Full 256-bit target | `{target_hex}` |
| Difficulty | `difficulty_1_target / target = {difficulty:,.2f}` |

A miner must hash the 80-byte block header (varying the nonce field)
until the double-SHA256 result is numerically smaller than this target.
            """
        )

    st.divider()

    # ------------------------------------------------------------------
    # Section 2 — Inter-arrival time histogram
    # ------------------------------------------------------------------
    st.subheader("⏱️ Block Inter-Arrival Time Distribution")

    timestamps = sorted([b["timestamp"] for b in blocks])
    inter_arrivals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    if len(inter_arrivals) < 5:
        st.warning("Not enough blocks to plot histogram.")
    else:
        arr = np.array(inter_arrivals, dtype=float)
        mean_t = arr.mean()
        lambda_hat = 1.0 / mean_t  # MLE for exponential distribution

        # Plotly histogram + theoretical PDF overlay
        x_range = np.linspace(0, max(arr.max(), EXPECTED_BLOCK_TIME * 4), 400)
        pdf_values = lambda_hat * np.exp(-lambda_hat * x_range)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=arr,
                nbinsx=30,
                histnorm="probability density",
                name="Observed",
                marker_color="#3d9be9",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf_values,
                mode="lines",
                name=f"Exponential PDF (λ=1/{mean_t:.0f}s)",
                line=dict(color="#f0383c", width=2.5, dash="dash"),
            )
        )
        fig.add_vline(
            x=EXPECTED_BLOCK_TIME,
            line_dash="dot",
            line_color="#ffd700",
            annotation_text="Expected 600 s",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Block Inter-Arrival Time Distribution (seconds)",
            xaxis_title="Inter-arrival time (seconds)",
            yaxis_title="Probability density",
            legend=dict(x=0.7, y=0.9),
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean inter-arrival", f"{mean_t:.1f} s")
        col_b.metric("Median inter-arrival", f"{np.median(arr):.1f} s")
        col_c.metric("Std dev", f"{arr.std():.1f} s")

        st.caption(
            "**Why exponential?** Mining is a memoryless Poisson process: each "
            "hash attempt succeeds independently with probability p ≈ target/2²⁵⁶. "
            "By the memoryless property (geometric → exponential in continuous time), "
            "waiting times between blocks follow Exp(λ = 1/600 s)."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Section 3 — Estimated network hash rate
    # ------------------------------------------------------------------
    st.subheader("💻 Estimated Network Hash Rate")

    # Formula: each valid hash is one "solution" to the PoW puzzle.
    # Expected attempts per block = 2^32 × difficulty (because difficulty
    # is normalised so that difficulty=1 requires 2^32 expected hashes).
    # Hash rate = expected_hashes_per_block / expected_time_per_block.
    hash_rate_hps = difficulty * (2 ** 32) / EXPECTED_BLOCK_TIME
    hash_rate_ehs = hash_rate_hps / 1e18  # convert to EH/s

    st.metric(
        "Estimated Hash Rate",
        f"{hash_rate_ehs:.2f} EH/s",
        help=(
            "Calculated as: difficulty × 2³² / 600 seconds. "
            "1 EH/s = 10¹⁸ hashes per second."
        ),
    )
    st.caption(
        f"Formula: `hash_rate ≈ {difficulty:,.0f} × 2³² / 600 "
        f"= {hash_rate_hps:.2e} H/s = {hash_rate_ehs:.2f} EH/s`"
    )
