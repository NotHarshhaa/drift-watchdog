# drift-watchdog 🐕

> Lightweight ML model drift detection — CLI, Prometheus metrics, and alerts. No platform required.

[![PyPI version](https://img.shields.io/pypi/v/drift-watchdog)](https://pypi.org/project/drift-watchdog/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange)](https://prometheus.io/)

---

## The problem

Your model was accurate last month. Now it's quietly wrong — and you don't know why.

Input distributions shift, upstream data pipelines change schema, feature encodings drift. Most teams have no drift detection at all, or rely on heavyweight ML platforms that take weeks to set up. **drift-watchdog fills that gap**: a single binary or Python sidecar that monitors your model's input/output distributions and fires alerts when something goes wrong.

---

## Features

- **Statistical drift detection** — PSI, KS-test, Jensen-Shannon divergence, and Wasserstein distance out of the box
- **Concept drift detection** — monitor model outputs and label distributions for performance degradation
- **HTML report export** — generate beautiful, shareable HTML reports for drift analysis
- **Prometheus exporter** — exposes `/metrics` endpoint, plug straight into your existing Grafana stack
- **CLI first** — run ad-hoc drift checks in CI/CD or cron without writing any code
- **Alert integrations** — Slack, PagerDuty, and webhook support
- **Framework agnostic** — works with scikit-learn, XGBoost, PyTorch, TensorFlow, or any model that takes tabular input
- **Reference baseline management** — store, version, and compare against baselines in local files, S3, or GCS
- **Lightweight** — no database, no server, no orchestrator required

---

## Quickstart

```bash
pip install drift-watchdog
```

### 1. Capture a reference baseline

```bash
drift-watchdog baseline create \
  --data reference_data.csv \
  --output baselines/v1.json \
  --name "production-v1"
```

### 2. Run a drift check

```bash
drift-watchdog check \
  --baseline baselines/v1.json \
  --current current_batch.csv \
  --threshold 0.2
```

```
✓ feature: age           PSI=0.04  [OK]
✓ feature: income        PSI=0.09  [OK]
⚠ feature: loan_amount   PSI=0.31  [DRIFT DETECTED]
✗ feature: credit_score  PSI=0.58  [SEVERE DRIFT]

Overall drift score: 0.43 — ALERT
```

### 3. Run as a Prometheus exporter

```bash
drift-watchdog serve \
  --baseline baselines/v1.json \
  --data-source s3://my-bucket/inference-logs/ \
  --port 9090 \
  --interval 300
```

Metrics are now available at `http://localhost:9090/metrics`.

### 4. Generate HTML report

```bash
drift-watchdog check \
  --baseline baselines/v1.json \
  --current current_batch.csv \
  --threshold 0.2 \
  --report drift_report.html
```

This generates a beautiful, shareable HTML report with detailed drift analysis.

### 5. Concept drift detection

Monitor model outputs and label distributions for performance degradation:

```bash
drift-watchdog concept-check \
  --baseline-predictions baseline_preds.csv \
  --baseline-labels baseline_labels.csv \
  --current-predictions current_preds.csv \
  --current-labels current_labels.csv \
  --threshold 0.2 \
  --report concept_drift_report.html
```

---

## Python API

```python
from drift_watchdog import DriftDetector, BaselineStore

store = BaselineStore("baselines/v1.json")
detector = DriftDetector(baseline=store.load())

result = detector.check(current_df)

for feature, report in result.features.items():
    print(f"{feature}: PSI={report.psi:.3f}, drift={report.is_drift}")

if result.overall_drift:
    result.alert()  # fires configured alert channels
```

---

## Configuration

Create a `watchdog.yaml` in your project root:

```yaml
baseline:
  path: baselines/v1.json
  storage: s3                      # local | s3 | gcs
  bucket: my-model-baselines

detection:
  methods: [psi, ks_test]
  thresholds:
    psi: 0.2                       # 0.1 = slight, 0.2 = moderate, 0.25+ = severe
    ks_pvalue: 0.05
  features:
    exclude: [id, timestamp]       # columns to skip

alerts:
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#ml-alerts"
  pagerduty:
    routing_key: ${PD_ROUTING_KEY}
    severity: warning
  webhook:
    url: https://your-endpoint.com/drift-event

exporter:
  port: 9090
  interval_seconds: 300
```

---

## Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `drift_watchdog_psi` | Gauge | PSI score per feature |
| `drift_watchdog_ks_statistic` | Gauge | KS-test statistic per feature |
| `drift_watchdog_feature_drift` | Gauge | 1 if drift detected, 0 if not |
| `drift_watchdog_overall_drift` | Gauge | 1 if any feature is drifting |
| `drift_watchdog_check_duration_seconds` | Histogram | Time taken per drift check |
| `drift_watchdog_last_check_timestamp` | Gauge | Unix timestamp of last check |

All metrics carry `feature`, `model`, and `baseline_version` labels.

---

## Kubernetes deployment

Run drift-watchdog as a sidecar alongside your model serving pod:

```yaml
# drift-watchdog-sidecar.yaml
containers:
  - name: drift-watchdog
    image: ghcr.io/your-org/drift-watchdog:latest
    args:
      - serve
      - --baseline
      - /baselines/v1.json
      - --data-source
      - $(INFERENCE_LOG_PATH)
      - --port
      - "9090"
    env:
      - name: SLACK_WEBHOOK_URL
        valueFrom:
          secretKeyRef:
            name: drift-watchdog-secrets
            key: slack-webhook
    ports:
      - containerPort: 9090
        name: metrics
```

Add the pod annotation and Prometheus will scrape it automatically:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
```

---

## Grafana dashboard

Import the pre-built dashboard from [`dashboards/drift-watchdog.json`](dashboards/drift-watchdog.json).

It includes panels for:
- Per-feature PSI over time
- Drift event timeline
- Feature distribution histograms (current vs baseline)
- Alert history

---

## Detection methods

| Method | Best for | Threshold guidance |
|---|---|---|
| **PSI** (Population Stability Index) | Categorical and continuous features | < 0.1 stable, 0.1–0.2 monitor, > 0.2 alert |
| **KS test** | Continuous distributions | p-value < 0.05 signals drift |
| **Jensen-Shannon divergence** | Probability distributions | > 0.1 worth alerting |
| **Wasserstein distance** | Ordinal/numeric features | Domain-dependent |
| **Chi-squared test** | Categorical features | p-value < 0.05 |

---

## Roadmap

- [x] **v1.0** — CLI, PSI + KS detection, local/S3/GCS baselines, Slack/PagerDuty/webhook alerts, Prometheus exporter, Grafana dashboard, Kubernetes sidecar example, watchdog.yaml config
- [x] **v1.1** — Concept drift detection (output/label distribution monitoring), HTML report export
- [ ] **v1.2** — GitHub Actions integration, CI drift gate
- [ ] **v1.3** — Multi-model support

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/your-username/drift-watchdog
cd drift-watchdog
pip install -e ".[dev]"
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).