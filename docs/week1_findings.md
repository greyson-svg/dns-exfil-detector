# Week 1 Findings — CIC-Bell-DNS-EXF 2021

One-page summary of what the EDA revealed. The supporting analysis lives in [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).

## Dataset snapshot

| View | Rows | Data columns | Notes |
|---|---|---|---|
| Stateless (per-query) | 757,211 | 15 | One row per DNS query. Primary input for the Week 2 XGBoost baseline. |
| Stateful (aggregated) | 262,105 | 27 | Pre-aggregated flow/window features. Useful as supplementary signal. |

**Class balance (stateless):** 462,858 benign vs 294,353 exfil → ~1.6:1 benign-to-exfil. Healthy for a security dataset — most are 100:1 or worse. No aggressive resampling needed for the baseline.

**Scenario coverage:**

- Light attacks: 42,683 stateless rows (slow-and-low exfil)
- Heavy attacks: 251,670 stateless rows (high-volume bursts)
- Benign (3 flavors): standalone 221K, matched-heavy 182K, matched-light 60K

The heavy/light split lets us test in Week 5 whether the per-query baseline misses slow-and-low cases (the entire motivation for the LSTM in Week 4).

**Payload types in attack data:** text (74K rows), audio (53K), compressed (46K), video (42K), exe (41K), image (37K). Reasonably balanced across the six categories — no single payload type dominates the attack signal.

## Promising features for the Week 2 baseline

Ranked by single-feature ROC AUC (how well one feature alone separates exfil from benign). Numbers above ~0.7 are useful; anything > ~0.99 would be a leakage red flag (none found).

**Per-query (stateless):**

| Feature | AUC | Why it makes sense |
|---|---|---|
| FQDN_count | 0.782 | Exfil queries pack many labels to smuggle payload chunks |
| numeric | 0.782 | Encoded payloads (base32 / base64-ish) carry digits; normal hostnames rarely do |
| subdomain_length | 0.776 | Encoded payload chunks live in the subdomain; longer = more payload |
| special | 0.770 | Special characters appear in encoded payloads more than in normal names |
| labels | 0.767 | Like FQDN_count — payload split across many labels |
| subdomain (binary) | 0.705 | Whether a subdomain is present at all |

**Aggregated (stateful):**

| Feature | AUC | Why it makes sense |
|---|---|---|
| A_frequency | 0.851 | Benign DNS is dominated by A-record lookups; this dataset's exfil isn't |
| rr | 0.851 | Likely a count tied to the same A-record signal |
| PTR_frequency | 0.830 | This dataset's exfil tools use PTR-record tunneling (see leakage notes) |
| ttl_mean | 0.824 | Exfil servers often use atypical TTLs |
| rr_name_length | 0.820 | Same intuition as subdomain_length but aggregated |
| rr_name_entropy | 0.812 | Encoded payloads have flatter character distributions = higher entropy |

The stateless features will be the bread and butter of Week 2. The stateful features are useful supplementary inputs but come with caveats (see below).

## Leakage analysis

### Smell test 1 — matched-benign vs standalone-benign (PASSED, with caveat)

The dataset contains benign traffic captured in three different conditions: alongside light attacks, alongside heavy attacks, and standalone. If the matched-benign captures were contaminated with testbed artifacts, their feature distributions would differ from the standalone benigns and we'd need to train on the standalone-only.

Result: on `subdomain_length`, `entropy`, and `FQDN_count`, the three benign scenarios have **essentially identical** means (within 1%) and **identical medians**. The matched benigns are not meaningfully different from the standalone benigns on these features. We can train on all benign data.

**Caveat:** this was checked on three features. A more rigorous version would compare distributions on all 27 stateful + 15 stateless features (and test for distributional differences, not just means). Worth doing if we see any model behavior that looks like benign-source leakage later.

### Smell test 2 — single-feature AUC > 0.99 (PASSED)

No single feature has AUC > 0.99. Highest is `A_frequency` at 0.85 (stateful) and `FQDN_count` at 0.78 (stateless). This means no feature directly encodes the label — the model will have to learn from feature combinations, which is what we want.

### Real concern: `rr_type` is a testbed-flavored signal

The `rr_type` column reveals that **93.75% of exfil rows involve PTR records, while 70.95% of benign rows are A records**. PTR-record tunneling is one form of DNS exfiltration, and the CIC team's tools used PTR. But real-world exfil tools use any record type — TXT and NULL are arguably more common in the wild.

**Implication for Week 2:** features that lean on record-type frequencies (`A_frequency`, `PTR_frequency`, `rr_type`) will work great on this dataset and **likely fail on the Week 5 self-generated dnscat2 / iodine traffic** — iodine uses NULL records by default, dnscat2 uses TXT. This is the cleanest "does the model generalize" test we'll have.

We should keep these features in the baseline (they're not strictly leakage), but **expect them to dominate** the feature importance, and **expect a generalization gap** in Week 5 driven by exactly this.

### Feature blocklist for Week 2

Do **NOT** include these as model inputs:

- `timestamp` — directly encodes capture session
- `source_file` — metadata, not a real feature
- `scenario`, `payload_type` — these are derived from the label path; using them would be label leakage
- Anything derived from source IP, resolver IP, or capture session (none present in these CSVs — the loader doesn't expose them)

## Data quality

| Check | Stateless | Stateful |
|---|---|---|
| Missing values (data cols) | longest_word: 31 rows | none |
| Duplicate rows | 526 / 757K = 0.07% | 148,760 / 262K = 56.76% |
| Negative entropy | 0 | n/a |
| Negative lengths | 0 | n/a |
| Value ranges | sensible (entropy 0.22–4.37, lengths 0–35) | sensible |

**Stateless: clean.** Drop the 31 longest_word nulls if it matters; replace with the median if not.

**Stateful: 57% duplicates is a problem.** Two windows producing identical aggregated features is plausible (both quiet periods of all-zero counts), but 57% is too high to ignore. For Week 2, deduplicate before training and report effective sample size. The duplicates also inflate measured AUCs — the "real" stateful single-feature AUCs may be a touch lower after dedup.

## Useless stateful columns

11 of 27 stateful columns have single-feature AUC = 0.500 because they're (near-)zero across both classes:

```
TXT_frequency, NULL_frequency, HINFO_frequency, MX_frequency,
NS_frequency, SOA_frequency, CNAME_frequency, AAAA_frequency,
SRV_frequency, OPT_frequency, a_records
```

These should be dropped before training — they contribute no signal and slow down model fitting. Note: this means the *interesting* record types for tunneling (TXT, NULL) carry no signal *in this dataset* because CIC's tools didn't use them. Real-world data would look different. Worth keeping the column-dropping logic configurable for that reason.

## Recommended Week 2 baseline plan

1. **Loader:** reuse the EDA notebook's `classify_file()` + `load_csvs()` logic; move it into `src/dns_exfil/data/loader.py`.
2. **Feature set (initial):** the 12 numeric stateless features minus `timestamp`. Add a handful of stateful features later as comparison: `rr_name_length`, `rr_name_entropy`, `ttl_mean`, `distinct_ns`.
3. **Train/test split:** stratify by `scenario` (not just label), so we don't accidentally train on all "heavy" and test on all "light" — that would inflate the test score.
4. **Model:** XGBoost classifier with default hyperparameters as the v0 reference number. Then a small grid over `max_depth`, `n_estimators`, `learning_rate`.
5. **Evaluation:** ROC AUC, PR AUC (more informative when classes are imbalanced), and confusion matrix at a chosen threshold. Report per-scenario breakdown so we see if the model is failing on light specifically (would justify the LSTM).
6. **Generalization safety net:** before declaring victory, retrain with `A_frequency`, `PTR_frequency`, `rr_type` REMOVED, and see how much performance drops. The gap is our Week 5 prediction interval.

## Open questions for Week 2

- What does `subdomain` mean as a numeric feature (range 0–1)? Probably a binary "has subdomain" flag, but the column header isn't explicit. Confirm by inspecting raw values.
- Are the 57% stateful duplicates true duplicates or artifacts of the aggregation window? Investigate before deduplication strategy.
- Should we compute our own per-query features from scratch (re-derive from pcaps) rather than trust CIC's? This is a Week 2 decision point — the answer affects whether we download pcaps.
