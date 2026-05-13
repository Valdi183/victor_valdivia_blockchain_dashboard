"""
Module M5 — Merkle Proof Verifier

Reconstructs a Bitcoin block's Merkle tree from scratch using only hashlib,
then verifies the inclusion of any selected transaction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bitcoin Merkle Tree — background
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Merkle tree is a binary hash tree where:
  • Leaves  : each txid in internal (little-endian) byte order.
  • Internal: SHA256(SHA256(left_child || right_child))
  • Root    : single hash stored in the block header as `merkle_root`.

Property: to prove tx_i is in a block with N transactions, only
⌈log₂(N)⌉ sibling hashes are needed — not all N txids.

BYTE ORDER
──────────
Txids from block explorers are in display (big-endian) hex.
Bitcoin hashes them in REVERSED byte order (internal/little-endian).
Each txid must be reversed before use as a leaf.

DUPLICATION RULE
────────────────
If a level has an odd number of hashes, the last hash is duplicated
to make an even count before pairing. This is a quirk of the original
Bitcoin Core implementation (src/consensus/merkle.cpp).

Reference: Bitcoin Core src/consensus/merkle.cpp
"""

import hashlib
import struct

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import (
    BlockchainAPIError,
    get_block_txids,
    get_latest_block,
    get_raw_header,
)


# ---------------------------------------------------------------------------
# Cryptographic helpers
# ---------------------------------------------------------------------------

def _dhash(data: bytes) -> bytes:
    """Double-SHA256: the standard hash function used in all Bitcoin structures."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def build_merkle_tree(txids_display: list[str]) -> tuple[list[list[bytes]], bytes]:
    """
    Build the complete Merkle tree from txids in display (big-endian) order.

    Algorithm
    ---------
    1. Reverse each txid's bytes → internal (little-endian) leaf hashes.
    2. While more than one hash remains:
       a. If the count is odd, duplicate the last hash (Bitcoin padding rule).
       b. Pair adjacent hashes: parent = SHA256(SHA256(left || right)).
    3. The single remaining hash is the Merkle root.

    Returns
    -------
    levels : list of lists; levels[0] = leaves, levels[-1] = [root].
             Each hash is 32 bytes in internal order.
    root   : Merkle root, 32 bytes, internal order.
    """
    # Step 1: convert display txids to internal byte order.
    current: list[bytes] = [bytes.fromhex(txid)[::-1] for txid in txids_display]
    levels: list[list[bytes]] = [current[:]]

    while len(current) > 1:
        # Step 2a: pad odd levels (Bitcoin duplication rule).
        if len(current) % 2 == 1:
            current = current + [current[-1]]

        # Step 2b: hash pairs.
        current = [
            _dhash(current[i] + current[i + 1])
            for i in range(0, len(current), 2)
        ]
        levels.append(current[:])

    return levels, current[0]


def get_merkle_proof(
    levels: list[list[bytes]], tx_index: int
) -> list[tuple[bytes, str]]:
    """
    Extract the Merkle authentication path for the transaction at tx_index.

    The proof is a sequence of (sibling_hash, side) pairs, one per level
    from the leaf up to (but not including) the root.

    Verification rule:
      side == 'right'  →  parent = dhash(our_node || sibling)
      side == 'left'   →  parent = dhash(sibling   || our_node)
    Applying all steps must reproduce the Merkle root.
    """
    proof: list[tuple[bytes, str]] = []
    idx = tx_index

    for level in levels[:-1]:
        # Apply the same padding used during tree construction.
        padded = level + [level[-1]] if len(level) % 2 == 1 else level

        if idx % 2 == 0:          # our node is the left child
            sibling_idx = idx + 1
            side = "right"
        else:                      # our node is the right child
            sibling_idx = idx - 1
            side = "left"

        proof.append((padded[sibling_idx], side))
        idx //= 2

    return proof


def verify_proof(
    txid_display: str,
    proof: list[tuple[bytes, str]],
    expected_root_internal: bytes,
) -> tuple[bool, list[str]]:
    """
    Re-derive the Merkle root from a txid + proof path and compare it to
    the block header's merkle_root field.

    Returns (is_valid, step_descriptions) — step_descriptions shows every
    hash concatenation so the computation is fully transparent.
    """
    node = bytes.fromhex(txid_display)[::-1]   # txid → internal order
    steps = [f"leaf  : {node[::-1].hex()[:20]}… (txid reversed to internal order)"]

    for i, (sibling, side) in enumerate(proof):
        if side == "right":
            combined = node + sibling
            label = f"L{i}→L{i+1}: dhash( our_node || sibling_right )"
        else:
            combined = sibling + node
            label = f"L{i}→L{i+1}: dhash( sibling_left || our_node  )"
        node = _dhash(combined)
        steps.append(f"{label} = {node[::-1].hex()[:20]}…")

    steps.append(f"computed root : {node[::-1].hex()}")
    steps.append(f"header  root  : {expected_root_internal[::-1].hex()}")
    return node == expected_root_internal, steps


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_proof_path(
    levels: list[list[bytes]], tx_index: int, proof: list[tuple[bytes, str]]
) -> None:
    """
    Plot the authentication path from leaf to root.

    Only O(log N) nodes are rendered, keeping the chart fast for large blocks.
    Layout:
      • Y axis = tree level (0 = leaf, top = root)
      • X = 0 → our node (red at leaf, green at derived nodes)
      • X = ±1 → sibling node (yellow; side determines sign)
    """
    fig = go.Figure()
    n_levels = len(levels)

    # Recompute our hash at each level so we can show the derived values.
    path_hashes: list[bytes] = []
    idx = tx_index
    for lvl_idx, level in enumerate(levels):
        padded = level + [level[-1]] if len(level) % 2 == 1 else level
        eff_idx = min(idx, len(padded) - 1)
        path_hashes.append(padded[eff_idx])
        idx //= 2

    for lvl_idx, (our_hash, entry) in enumerate(zip(path_hashes, proof + [None])):  # type: ignore[list-item]
        y = lvl_idx

        # Our node
        color = "#f0383c" if lvl_idx == 0 else "#4caf50"
        label = ("★ tx" if lvl_idx == 0 else "derived") + f" {our_hash[::-1].hex()[:10]}…"
        fig.add_trace(go.Scatter(
            x=[0], y=[y], mode="markers+text",
            marker=dict(size=16, color=color, line=dict(width=1, color="#666")),
            text=[label], textposition="middle right", textfont=dict(size=9),
            showlegend=False,
        ))

        if entry is not None:
            sibling, side = entry
            sx = 1 if side == "right" else -1
            sib_label = f"sib({side}) {sibling[::-1].hex()[:10]}…"
            fig.add_trace(go.Scatter(
                x=[sx], y=[y], mode="markers+text",
                marker=dict(size=14, color="#ffd700", line=dict(width=1, color="#666")),
                text=[sib_label],
                textposition="middle left" if sx > 0 else "middle right",
                textfont=dict(size=9),
                showlegend=False,
            ))
            # Line from our node and sibling up to parent
            parent_y = y + 1
            for x_start in [0, sx]:
                fig.add_shape(type="line", x0=x_start, y0=y, x1=0, y1=parent_y,
                              line=dict(color="rgba(255,255,255,0.25)", width=1))

    # Root node (top level)
    root_hash = levels[-1][0]
    fig.add_trace(go.Scatter(
        x=[0], y=[n_levels - 1], mode="markers+text",
        marker=dict(size=18, color="#00e5ff", symbol="diamond",
                    line=dict(width=1, color="#666")),
        text=[f"ROOT {root_hash[::-1].hex()[:12]}…"],
        textposition="middle right", textfont=dict(size=10),
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Merkle proof path — tx index {tx_index} ({n_levels - 1} levels to root)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-2.5, 2.5]),
        yaxis=dict(
            title="Tree level (0 = leaves)",
            tickmode="array",
            tickvals=list(range(n_levels)),
            ticktext=[
                f"L{i} — leaf (txids)" if i == 0
                else f"L{i} — root" if i == n_levels - 1
                else f"L{i}"
                for i in range(n_levels)
            ],
        ),
        template="plotly_dark",
        height=max(400, 70 * n_levels),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Selected tx · 🟡 Proof siblings · 🟢 Derived nodes · 🔵 Root")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the M5 Merkle Proof Verifier tab."""

    st.header("🌿 M5 — Merkle Proof Verifier")
    st.caption(
        "Reconstruct a Bitcoin block's Merkle tree from scratch with `hashlib` "
        "and verify any transaction's inclusion proof step by step."
    )

    # ------------------------------------------------------------------
    # Block input
    # ------------------------------------------------------------------
    col_inp, col_btn = st.columns([3, 1])
    with col_inp:
        block_hash_input = st.text_input(
            "Block hash (leave blank for latest block)",
            placeholder="000000000000000000…",
            key="m5_block_hash",
        )
    with col_btn:
        st.write("")
        if st.button("Use latest block", key="m5_latest_btn"):
            try:
                latest = get_latest_block()
                block_hash_input = latest["id"]
                st.session_state["m5_forced_hash"] = block_hash_input
            except BlockchainAPIError as exc:
                st.error(f"Could not fetch latest block: {exc}")
                return

    # Use the forced hash from the button if the text input is empty
    if not block_hash_input:
        block_hash_input = st.session_state.get("m5_forced_hash", "")
    if not block_hash_input:
        try:
            latest = get_latest_block()
            block_hash_input = latest["id"]
        except BlockchainAPIError as exc:
            st.error(f"Could not fetch latest block: {exc}")
            return

    # ------------------------------------------------------------------
    # Fetch txids and raw header
    # ------------------------------------------------------------------
    try:
        with st.spinner("Fetching block data…"):
            txids = get_block_txids(block_hash_input)
            raw_header = get_raw_header(block_hash_input)
    except BlockchainAPIError as exc:
        st.error(f"API error: {exc}")
        return

    n_tx = len(txids)
    if n_tx == 0:
        st.error("No transactions found in this block.")
        return

    st.success(f"Block `{block_hash_input[:24]}…` — **{n_tx:,} transactions**")

    # ------------------------------------------------------------------
    # Transaction selector
    # ------------------------------------------------------------------
    st.subheader("🔎 Select a Transaction")
    col_idx, col_display = st.columns([1, 3])
    with col_idx:
        tx_index = int(st.number_input(
            "Transaction index (0 = coinbase)",
            min_value=0, max_value=n_tx - 1, value=0, step=1, key="m5_tx_index",
        ))
    with col_display:
        st.text_input("Selected txid", value=txids[tx_index], disabled=True,
                      key="m5_txid_disp")

    if not st.button("▶ Build Merkle tree & generate proof", key="m5_run"):
        st.info("Select a transaction and click the button to run the computation.")
        return

    # ------------------------------------------------------------------
    # Build the Merkle tree (core cryptographic computation)
    # ------------------------------------------------------------------
    with st.spinner(f"Building Merkle tree for {n_tx:,} transactions…"):
        levels, computed_root = build_merkle_tree(txids)

    # Extract merkle_root from raw header: bytes 36–68 in internal order.
    # The raw header is little-endian; byte-reverse for display comparison.
    merkle_root_internal = raw_header[36:68]

    # ------------------------------------------------------------------
    # Section 1 — Tree structure summary
    # ------------------------------------------------------------------
    st.subheader("🌳 Tree Structure Overview")

    level_rows = []
    for i, lvl in enumerate(levels):
        level_rows.append({
            "Level": i,
            "Role": "Leaves (txids)" if i == 0 else "Root" if i == len(levels) - 1 else f"Internal L{i}",
            "Hashes": len(lvl),
            "Sample hash (display order)": lvl[0][::-1].hex()[:32] + "…",
        })
    st.dataframe(pd.DataFrame(level_rows), use_container_width=True, hide_index=True)

    # Tree shape as a horizontal bar chart (shows the pyramid structure).
    fig_tree = go.Figure(go.Bar(
        y=[f"L{i}" for i in range(len(levels))],
        x=[len(lvl) for lvl in levels],
        orientation="h",
        marker_color="#3d9be9",
        hovertemplate="Level %{y}: %{x} hashes<extra></extra>",
    ))
    fig_tree.update_layout(
        title="Merkle tree pyramid — hashes per level",
        xaxis_title="Number of hashes",
        yaxis_title="Level",
        template="plotly_dark",
        height=max(250, 40 * len(levels)),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 2 — Root verification
    # ------------------------------------------------------------------
    st.subheader("✅ Merkle Root Verification")

    roots_match = computed_root == merkle_root_internal
    col_c, col_h = st.columns(2)
    col_c.markdown("**Computed root (this module):**")
    col_c.code(computed_root[::-1].hex())
    col_h.markdown("**Header `merkle_root` field:**")
    col_h.code(merkle_root_internal[::-1].hex())

    if roots_match:
        st.success(
            "✅ Computed Merkle root matches the block header — "
            "all transactions are correctly committed."
        )
    else:
        st.error("❌ Root mismatch. Check byte-order handling or API response.")

    # ------------------------------------------------------------------
    # Section 3 — Inclusion proof for the selected transaction
    # ------------------------------------------------------------------
    st.subheader(f"🔐 Inclusion Proof — Transaction #{tx_index}")

    proof = get_merkle_proof(levels, tx_index)
    is_valid, steps = verify_proof(txids[tx_index], proof, merkle_root_internal)

    # Proof path table
    proof_rows = [
        {
            "Step": i + 1,
            "From level": i,
            "Sibling side": side,
            "Sibling hash (display)": h[::-1].hex()[:40] + "…",
        }
        for i, (h, side) in enumerate(proof)
    ]
    if proof_rows:
        st.dataframe(pd.DataFrame(proof_rows), use_container_width=True, hide_index=True)
    else:
        st.info("This is the only transaction (single-tx block) — no siblings needed.")

    # Step-by-step hash computation
    st.markdown("**Step-by-step re-derivation (every hash concatenation):**")
    for step in steps:
        st.code(step, language="text")

    if is_valid:
        st.success(f"✅ Proof valid — transaction at index {tx_index} is confirmed in this block.")
    else:
        st.error("❌ Proof invalid.")

    # ------------------------------------------------------------------
    # Section 4 — Proof path visualisation
    # ------------------------------------------------------------------
    st.subheader("📊 Proof Path Visualisation")
    _plot_proof_path(levels, tx_index, proof)

    with st.expander("📖 Merkle trees and why they matter"):
        st.markdown(
            rf"""
**Merkle tree** (Ralph Merkle, 1979) is a binary hash tree used in Bitcoin to
commit to all transactions in a block with a single 32-byte root.

**Inclusion proof** for txᵢ in a block with N transactions:
- Requires only `⌈log₂(N)⌉` sibling hashes — **not** all N txids.
- For this block: {n_tx} txids → proof length = **{len(proof)} hashes**
  ({len(proof)*32} bytes vs {n_tx*32:,} bytes for the full txid list).

**Byte order rule:** Bitcoin reverses txid bytes before hashing (internal order).
Mixing display and internal order is the most common implementation bug.

**Duplication rule:** if a level has an odd number of nodes, the last is
duplicated — this is a Bitcoin-specific quirk with no mathematical motivation.

**Verification:** apply each (sibling, side) step to re-derive the root.
If the result equals the header's `merkle_root`, the transaction is proven
to be in the block **without trusting the API**.
            """
        )
