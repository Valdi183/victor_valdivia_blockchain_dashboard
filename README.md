# CryptoChain Analyzer Dashboard

## Student Information

| Field | Value |
|---|---|
| Name | Victor Valdivia Calatrava |
| GitHub Username | Valdi183 |
| Course | Cryptography — Universidad Alfonso X el Sabio |
| Professor | Jorge Calvo — Academic Year 2025–26 |

## Project Description

CryptoChain Analyzer is a real-time Bitcoin blockchain dashboard built with Streamlit that fetches live data from the Blockstream and Mempool.space public APIs. It implements seven modules covering core cryptographic concepts (Proof of Work, block header parsing, Merkle trees, difficulty adjustment, 51% attack cost) and two independent AI approaches (statistical anomaly detection and supervised fee-rate prediction). The dashboard auto-refreshes every 60 seconds and falls back gracefully to cached data on API failure.

## Chosen AI Approaches

- **M4:** Unsupervised statistical anomaly detector — fits an Exponential(λ) model to Bitcoin inter-arrival times by MLE and flags blocks whose survival probability falls below 1% as statistically rare events.
- **M7:** Supervised Gradient Boosting Regressor — predicts the next period's median transaction fee (sat/vByte) from lag features and time-of-day patterns derived from mempool.space fee rate history.

## Module Status

| Module | Title | Status |
|--------|-------|--------|
| M1 | Proof of Work Monitor | ✅ Complete |
| M2 | Block Header Analyzer | ✅ Complete |
| M3 | Difficulty History | ✅ Complete |
| M4 | AI Component — Anomaly Detector | ✅ Complete |
| M5 | Merkle Proof Verifier | ✅ Complete |
| M6 | Security Score (51% Attack Cost) | ✅ Complete |
| M7 | Second AI Approach — Fee Estimator | ✅ Complete |

✅ Project completed

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
template-blockchain-dashboard/
├── app.py                        # Streamlit entry point (7 tabs, auto-refresh 60 s)
├── requirements.txt
├── README.md
├── api/
│   └── blockchain_client.py      # HTTP client — Blockstream + Mempool.space APIs
├── modules/
│   ├── m1_proof_of_work.py       # bits decoding, difficulty, inter-arrival histogram
│   ├── m2_block_header.py        # 80-byte header parser, local PoW verification
│   ├── m3_difficulty_history.py  # 2-year difficulty chart, DAA epoch markers
│   ├── m4_ai_component.py        # Exponential MLE anomaly detector (KS evaluation)
│   ├── m5_merkle_proof.py        # Full Merkle tree + inclusion proof verifier
│   ├── m6_security_score.py      # 51% attack cost + Nakamoto §11 probability
│   └── m7_fee_estimator.py       # Gradient Boosting fee regression (MAE, RMSE, R²)
└── report/
    ├── report.md                 # Technical report (source)
    └── report.pdf                # Technical report (rendered)
```

## External References

Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System*. https://bitcoin.org/bitcoin.pdf

Blockstream API Documentation. *Blockstream Explorer API*. https://blockstream.info/api

Mempool.space API Documentation. *Mempool.space API*. https://mempool.space/api
