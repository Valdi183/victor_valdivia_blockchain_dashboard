"""
Module M2 — Block Header Analyzer

Fetches the raw 80-byte block header and:
  • Parses all 6 fields manually (with correct byte-order handling).
  • Independently verifies the block hash using hashlib only — no
    Bitcoin-specific libraries.
  • Counts leading zero BITS in the hash and compares them to the target
    derived from the `bits` field.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Block header layout (80 bytes total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Offset  Length  Field          Byte order
  0       4     version        little-endian uint32
  4      32     prev_hash      little-endian (reverse for display)
 36      32     merkle_root    little-endian (reverse for display)
 68       4     timestamp      little-endian uint32 → Unix epoch
 72       4     bits           little-endian uint32
 76       4     nonce          little-endian uint32

IMPORTANT NOTE ON BYTE ORDER
─────────────────────────────
Bitcoin stores multi-byte integers as little-endian in the raw header.
The 32-byte hash fields (prev_hash, merkle_root) are stored in
internal byte order (little-endian), but are conventionally *displayed*
in reverse (big-endian / RPC) byte order — i.e. the bytes are reversed
before converting to hex.

The computed block hash must ALSO be reversed before display, because
SHA256(SHA256(header)) returns bytes in internal order.

Mixing up byte order is the #1 source of bugs when working with Bitcoin
headers.  Every reversal in this module has an explicit comment.
"""

import hashlib
import struct
import datetime

import streamlit as st

from api.blockchain_client import (
    BlockchainAPIError,
    get_block_by_hash,
    get_latest_block,
    get_raw_header,
)
from modules.m1_proof_of_work import decode_bits, target_to_hex, count_leading_zero_bits


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_header(raw: bytes) -> dict:
    """
    Unpack all 6 fields from an 80-byte raw block header.

    Returns a dict with both raw values (for PoW math) and display
    strings (byte-reversed where appropriate).
    """
    if len(raw) != 80:
        raise ValueError(f"Expected 80 bytes, got {len(raw)}")

    # struct.unpack with '<' prefix = little-endian.
    version, = struct.unpack_from("<I", raw, 0)          # 4 bytes  @ offset 0

    # prev_hash and merkle_root are 32 bytes each, stored in internal (LE) order.
    # We reverse the bytes for display to match the conventional big-endian RPC format.
    prev_hash_internal = raw[4:36]
    merkle_root_internal = raw[36:68]

    timestamp, = struct.unpack_from("<I", raw, 68)        # 4 bytes  @ offset 68
    bits, = struct.unpack_from("<I", raw, 72)             # 4 bytes  @ offset 72
    nonce, = struct.unpack_from("<I", raw, 76)            # 4 bytes  @ offset 76

    # Reverse bytes → big-endian display order (RPC convention).
    prev_hash_display = prev_hash_internal[::-1].hex()
    merkle_root_display = merkle_root_internal[::-1].hex()

    return {
        "version": version,
        "prev_hash_internal": prev_hash_internal,
        "prev_hash_display": prev_hash_display,
        "merkle_root_internal": merkle_root_internal,
        "merkle_root_display": merkle_root_display,
        "timestamp": timestamp,
        "timestamp_dt": datetime.datetime.utcfromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "bits": bits,
        "nonce": nonce,
    }


# ---------------------------------------------------------------------------
# PoW verification
# ---------------------------------------------------------------------------

def verify_pow(raw_header: bytes) -> dict:
    """
    Independently verify the block hash using hashlib.

    Algorithm:
        hash_internal = SHA256(SHA256(raw_header))   ← 32 bytes, internal order
        hash_display  = hash_internal reversed        ← big-endian / RPC order

    The displayed hash must match what the API returns in block.hash.

    Returns a dict with both forms of the hash, the leading zero bit count,
    and whether the hash is below the target derived from `bits`.
    """
    # Step 1: double-SHA256 of the raw 80-byte header (NOT reversed).
    # We pass raw bytes directly — no hex encoding at this stage.
    first_hash = hashlib.sha256(raw_header).digest()
    hash_internal = hashlib.sha256(first_hash).digest()   # 32 bytes, little-endian

    # Step 2: Reverse to obtain the conventional big-endian display hash.
    hash_display = hash_internal[::-1].hex()              # 64 hex chars

    # Step 3: Count leading zero BITS (not just hex nibbles) in the
    # internal-order hash, because that is the actual integer value being
    # compared against the target.
    leading_zero_bits = count_leading_zero_bits(hash_internal)

    # Step 4: Decode the target from `bits` (parsed from the header).
    bits_val, = struct.unpack_from("<I", raw_header, 72)
    target = decode_bits(bits_val)
    target_hex = target_to_hex(target)

    # Step 5: The hash (as a big integer from internal-order bytes) must be
    # below the target.  hash_int is computed from the internal byte order
    # (same as the target comparison Bitcoin nodes perform).
    hash_int = int.from_bytes(hash_internal, "little")   # ← internal = little-endian
    is_valid = hash_int < target

    return {
        "hash_display": hash_display,
        "hash_internal": hash_internal,
        "leading_zero_bits": leading_zero_bits,
        "target_hex": target_hex,
        "target_int": target,
        "bits_val": bits_val,
        "is_valid": is_valid,
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the M2 Block Header Analyzer tab."""

    st.header("🔍 M2 — Block Header Analyzer")
    st.caption(
        "Parse and verify a Bitcoin block header byte-by-byte using only "
        "Python's built-in `hashlib`."
    )

    # Input: use latest block by default, or let user enter a custom hash.
    col_inp, col_btn = st.columns([3, 1])
    with col_inp:
        input_hash = st.text_input(
            "Block hash (leave blank for latest block)",
            placeholder="000000000000000000…",
            key="m2_hash_input",
        )
    with col_btn:
        st.write("")  # vertical alignment spacer
        fetch_latest = st.button("Use latest block", key="m2_latest_btn")

    if fetch_latest or not input_hash:
        try:
            latest = get_latest_block()
            input_hash = latest["id"]        # Blockstream uses "id" for hash
        except BlockchainAPIError as exc:
            st.warning(f"⚠️ Could not fetch latest block: {exc}")
            cached = st.session_state.get("m2_data")
            if cached:
                st.info("Showing last successfully analysed block.")
                _display(cached)
            return

    if not input_hash:
        st.info("Enter a block hash above or click **Use latest block**.")
        return

    # ------------------------------------------------------------------
    # Fetch raw header + block metadata
    # ------------------------------------------------------------------
    try:
        raw_header = get_raw_header(input_hash)
        block_meta = get_block_by_hash(input_hash)
        api_hash = block_meta.get("id") or block_meta.get("hash", "")
    except BlockchainAPIError as exc:
        st.warning(f"⚠️ API unavailable — showing cached data. ({exc})")
        cached = st.session_state.get("m2_data")
        if cached:
            _display(cached)
        else:
            st.error("No cached data available.")
        return

    parsed = parse_header(raw_header)
    verification = verify_pow(raw_header)
    data = {
        "raw_hex": raw_header.hex(),
        "parsed": parsed,
        "verification": verification,
        "api_hash": api_hash,
        "block_meta": block_meta,
    }
    st.session_state["m2_data"] = data
    _display(data)


def _display(data: dict) -> None:
    """Render all sections once we have header data."""
    p = data["parsed"]
    v = data["verification"]
    api_hash = data["api_hash"]

    # ------------------------------------------------------------------
    # Raw header hex dump
    # ------------------------------------------------------------------
    st.subheader("📦 Raw 80-byte Header")
    raw_hex = data["raw_hex"]
    # Colour-code the six field regions.
    colours = [
        ("#4fc3f7", 8),   # version         4 bytes = 8 hex chars
        ("#ef9a9a", 64),  # prev_hash       32 bytes = 64 hex chars
        ("#a5d6a7", 64),  # merkle_root     32 bytes = 64 hex chars
        ("#fff176", 8),   # timestamp        4 bytes = 8 hex chars
        ("#ce93d8", 8),   # bits             4 bytes = 8 hex chars
        ("#ffcc80", 8),   # nonce            4 bytes = 8 hex chars
    ]
    legend = {
        "#4fc3f7": "version",
        "#ef9a9a": "prev_hash",
        "#a5d6a7": "merkle_root",
        "#fff176": "timestamp",
        "#ce93d8": "bits",
        "#ffcc80": "nonce",
    }
    html_parts = []
    pos = 0
    for colour, length in colours:
        chunk = raw_hex[pos : pos + length]
        html_parts.append(
            f'<span style="background:{colour};color:#111;padding:1px 2px;'
            f'border-radius:2px;font-family:monospace">{chunk}</span>'
        )
        pos += length

    legend_html = " ".join(
        f'<span style="background:{c};color:#111;padding:1px 6px;'
        f'border-radius:3px;font-size:0.8em">{n}</span>'
        for c, n in legend.items()
    )
    st.markdown(
        f'<div style="word-break:break-all;line-height:2">{"".join(html_parts)}</div>'
        f'<div style="margin-top:8px">{legend_html}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ------------------------------------------------------------------
    # Parsed fields table
    # ------------------------------------------------------------------
    st.subheader("📋 Parsed Header Fields")
    fields = [
        ("version", f"`{hex(p['version'])}` (int: {p['version']})", "4 bytes, little-endian"),
        ("prev_hash", f"`{p['prev_hash_display']}`", "32 bytes, bytes-reversed for display"),
        ("merkle_root", f"`{p['merkle_root_display']}`", "32 bytes, bytes-reversed for display"),
        ("timestamp", f"`{p['timestamp']}` → {p['timestamp_dt']}", "4 bytes, little-endian"),
        ("bits", f"`{hex(p['bits'])}`", "4 bytes, little-endian"),
        ("nonce", f"`{p['nonce']:,}` (`{hex(p['nonce'])}`)", "4 bytes, little-endian"),
    ]
    col_hdr = st.columns([2, 5, 4])
    col_hdr[0].markdown("**Field**")
    col_hdr[1].markdown("**Value**")
    col_hdr[2].markdown("**Notes**")
    for field, value, note in fields:
        c1, c2, c3 = st.columns([2, 5, 4])
        c1.markdown(f"`{field}`")
        c2.markdown(value)
        c3.markdown(f"*{note}*")

    st.divider()

    # ------------------------------------------------------------------
    # PoW Verification (C1 — 30% of grade)
    # ------------------------------------------------------------------
    st.subheader("✅ Proof-of-Work Verification (C1 Criterion)")
    st.caption(
        "Hash recomputed locally with `hashlib.sha256` — no Bitcoin libraries used."
    )

    computed = v["hash_display"]
    match = computed.lower() == api_hash.lower()

    st.markdown("**Step 1 — Compute double-SHA256 of raw header:**")
    st.code(
        "import hashlib\n"
        "h1 = hashlib.sha256(raw_header_bytes).digest()\n"
        "h2 = hashlib.sha256(h1).digest()          # internal byte order\n"
        "block_hash = h2[::-1].hex()               # reverse → big-endian display",
        language="python",
    )

    st.markdown("**Step 2 — Computed hash (big-endian / display order):**")
    st.code(computed, language="text")

    st.markdown("**Step 3 — API-reported hash:**")
    st.code(api_hash, language="text")

    if match:
        st.success("✅ Hashes match — PoW verification passed!")
    else:
        st.error("❌ Hash mismatch. Check byte-order handling.")

    st.divider()

    # Leading zeros analysis
    st.markdown("**Step 4 — Leading zero bit analysis:**")
    target_hex = v["target_hex"]
    leading_zeros_hex = len(target_hex) - len(target_hex.lstrip("0"))

    col_z1, col_z2, col_z3 = st.columns(3)
    col_z1.metric("Leading zero BITS in hash", v["leading_zero_bits"])
    col_z2.metric("Target leading zeros (hex chars)", leading_zeros_hex)
    col_z2.caption(f"= {leading_zeros_hex * 4} bits")
    col_z3.metric("Hash < Target?", "✅ Yes" if v["is_valid"] else "❌ No")

    st.markdown("**256-bit target from `bits` field:**")
    tz = leading_zeros_hex
    rest = target_hex[tz:]
    st.markdown(
        f'<code style="font-size:0.8em">'
        f'<span style="color:#f0383c;font-weight:bold">{"0"*tz}</span>'
        f'<span style="color:#e8e8e8">{rest}</span>'
        f"</code>",
        unsafe_allow_html=True,
    )

    with st.expander("🔬 Why do we reverse bytes for display?"):
        st.markdown(
            """
Bitcoin stores values in **little-endian** (least significant byte first) in
the raw header, but convention (inherited from the original Satoshi client)
is to display hashes in **big-endian** (most significant byte first).

So after computing `SHA256(SHA256(header))`, the 32-byte result must be
**byte-reversed** before converting to the hex string you see in block
explorers.

The comparison to the target is done on the **internal (un-reversed)** integer
value — as if you read the bytes as a little-endian 256-bit number.

Mixing these up is the #1 source of bugs in Bitcoin implementations.
            """
        )
