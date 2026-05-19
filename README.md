# DNS Exfiltration Detector

ML system that flags when DNS is being used as a covert exfiltration or command-and-control channel.

## The problem

DNS is one of the few protocols that egresses from nearly every network unfiltered, which makes it a favorite carrier for covert data movement. Tools like `iodine`, `dnscat2`, and `DNSExfiltrator` smuggle payloads inside subdomain labels or TXT records — traffic that looks like ordinary DNS unless you know what to look for. This project detects that pattern from passive DNS observation.

## Approach: two models, increasing capability

| Stage | Model | What it sees | Catches |
|---|---|---|---|
| 1. Baseline | XGBoost | per-query features (length, Shannon entropy, char-dist, record type) | high-volume exfil where individual queries look anomalous |
| 2. Flow model | LSTM / small transformer | sequences of queries from one source over a time window | slow-and-low exfil that hides any single query in normal-looking noise |

The baseline establishes a strong reference number. The deep model only earns its keep if it beats the baseline on slow-and-low attacks (Week 5 generalization test).

## 6-week build plan

| Week | Deliverable | Status |
|---|---|---|
| 1 | EDA + dataset acquisition + project foundation | in progress |
| 2 | Per-query baseline (feature engineering + XGBoost) | planned |
| 3 | Char-level CNN on raw domain strings | planned |
| 4 | Flow-level temporal model (LSTM) | planned |
| 5 | Self-generated attack data (dnscat2/iodine), generalization test | planned |
| 6 | Deployment: FastAPI + Streamlit + Pi-hole integration | planned |

## Reproduce

Prerequisites: Python 3.12, [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/greyson-svg/dns-exfil-detector
cd dns-exfil-detector
uv sync                                    # creates .venv, installs from uv.lock
uv run jupyter lab notebooks/01_eda.ipynb  # run EDA notebook
```

## Data

This project uses **CIC-Bell-DNS-EXF 2021** from the University of New Brunswick's Canadian Institute for Cybersecurity. The dataset is not redistributable — you need to register on UNB's portal to download it.

1. Register at https://www.unb.ca/cic/datasets/dns-exf-2021.html
2. Receive download link via email
3. Place files in `data/raw/`
4. Sanity-check: `uv run python scripts/verify_dataset.py`

## Layout

```
src/dns_exfil/      # importable package
  data/             # loaders + schema definitions
  features/         # feature engineering (entropy, length, char-dist)
  utils/
notebooks/          # exploratory analysis (versioned)
tests/              # pytest suite
docs/               # weekly findings, methodology notes
scripts/            # one-off scripts (dataset verification, etc.)
data/               # gitignored — never committed
```

## Methodology notes

**Capture-provenance leakage.** CIC-Bell-DNS-EXF was built by capturing benign and exfiltration traffic on separate testbeds, then merging. Naive features like source IP, capture timestamp, or resolver TTL will leak the provenance — the model "learns" which capture file a row came from rather than what exfiltration actually looks like. See [`docs/week1_findings.md`](docs/week1_findings.md) for the explicit feature blocklist.

**Generalization test.** Week 5 retests the trained model on self-generated `dnscat2` / `iodine` traffic captured in a different environment. If accuracy collapses, the model was leaking. If it generalizes, the methodology held.

## Tech stack

Python 3.12 · `uv` · pandas · scikit-learn · XGBoost · PyTorch · pytest · ruff · FastAPI · Streamlit

## Author

Greyson Ballard — [@greyson-svg](https://github.com/greyson-svg)
