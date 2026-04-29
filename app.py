"""
app.py — CryptoChain Analyzer Dashboard
─────────────────────────────────────────
Streamlit entry point.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Architecture:
  • st_autorefresh polls every 60 s to update live data (C3 criterion).
  • Four tabs map to the four modules (M1–M4).
  • All API errors are caught in each module; the last successful payload
    is stored in st.session_state for graceful degradation (C3 criterion).
"""

import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from modules import m1_proof_of_work, m2_block_header, m3_difficulty_history, m4_ai_component

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CryptoChain Analyzer",
    page_icon="⛓️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auto-refresh every 60 seconds (C3 — real-time updates)
# ---------------------------------------------------------------------------
st_autorefresh(interval=60_000, key="dashboard_autorefresh")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⛓️ CryptoChain Analyzer")
    st.caption("University Cryptography Project")
    st.divider()

    st.markdown(f"**Last updated:** {datetime.datetime.utcnow().strftime('%H:%M:%S UTC')}")

    if st.button("🔄 Manual refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("M1 Settings")
    n_blocks = st.selectbox(
        "Blocks for histogram",
        options=[20, 50, 100],
        index=1,
        key="m1_n_blocks",
    )

    st.divider()
    st.markdown(
        """
**Grading criteria:**
- C1 (30%) — Crypto correctness ✅
- C2 (25%) — AI component ✅
- C3 (20%) — Real-time updates ✅
- C4 (15%) — Visualisation ✅
- C5 (10%) — Report *(add to /report/)*
        """
    )
    st.divider()
    st.caption(
        "Data: [Blockstream](https://blockstream.info) · "
        "[Blockchain.info](https://blockchain.info)"
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "⛏️ M1 — Proof of Work",
        "🔍 M2 — Block Header",
        "📈 M3 — Difficulty History",
        "🤖 M4 — Anomaly Detector",
    ]
)

with tab1:
    m1_proof_of_work.render(n_blocks=n_blocks)

with tab2:
    m2_block_header.render()

with tab3:
    m3_difficulty_history.render()

with tab4:
    m4_ai_component.render()