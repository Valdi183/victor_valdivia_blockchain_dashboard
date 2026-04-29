# Blockchain Dashboard Project

Use this repository to build your blockchain dashboard project.
Update this README every week.

## Student Information

| Field | Value |
|---|---|
| Student Name | Victor Valdivia Calatrava |
| GitHub Username | Valdi183 |
| Project Title | victor_valdivia_blockchain_dashboard |
| Chosen AI Approach | Option 2 "Anomaly detector"|

## Module Tracking

Use one of these values: `Not started`, `In progress`, `Done`

| Module | What it should include | Status |
|---|---|---|
| M1 | Proof of Work Monitor | Done |
| M2 | Block Header Analyzer | Done |
| M3 | Difficulty History | Done |
| M4 | AI Component | Done |

## Current Progress

Write 3 to 5 short lines about what you have already done.

I built a real-time Bitcoin blockchain dashboard using Streamlit, fetching live data from Blockstream and Blockchain.info through a custom API client with
retry logic and TTL caching. M1 decodes the compact bits field into a full 256-bit target and plots inter-arrival time distributions against a theoretical
exponential PDF. M2 independently verifies block hashes using only Python's hashlib, parsing all 6 raw header fields with correct byte-order handling. M3
charts two years of difficulty history with per-period adjustment markers, while M4 detects anomalous blocks by fitting an exponential distribution to inter
arrival times and flagging statistical outliers via p-values. The dashboard auto-refreshes every 60 seconds and falls back to cached data gracefully on API
failure.

## Next Step

Write the next small step you will do before the next class.

- I am planning to incorporate more functionalities to the program

## Main Problem or Blocker

Write here if you are stuck with something.

- 

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
template-blockchain-dashboard/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- app.py
|-- api/
|   `-- blockchain_client.py
`-- modules/
    |-- m1_pow_monitor.py
    |-- m2_block_header.py
    |-- m3_difficulty_history.py
    `-- m4_ai_component.py
```

<!-- student-repo-auditor:teacher-feedback:start -->
## Teacher Feedback

### Kick-off Review

Review time: 2026-04-16 09:59 CEST
Status: Amber

Strength:
- Your repository keeps the expected classroom structure.

Improve now:
- The README is present but still misses part of the required kickoff information.

Next step:
- Complete the README fields for student information, AI approach, module status, and next step.
<!-- student-repo-auditor:teacher-feedback:end -->
