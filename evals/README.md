# Pure Mathematical Evaluations & Metrics

**Complete implementation of pure mathematical evaluation framework for transaction analysis agent.**

---

## Overview

This package implements deterministic, LLM-free evaluation metrics for measuring agent performance. All metrics are pure Python functions - no LLM-as-judge.

### Key Principles

1. ✅ **No LLM-as-judge** (REQ-EVAL-003): Every metric is a deterministic Python function
2. ✅ **Reproducible** (REQ-EVAL-002): Fixed random seed ensures consistent results
3. ✅ **Immutable** (REQ-METRIC-SEC-001): Evaluation results cannot be modified
4. ✅ **Auditable** (REQ-METRIC-SEC-002): Compliance export for 7-year retention
5. ✅ **Integrated** (REQ-LOGFIRE-METRIC-001): Full Logfire logging and dashboards

---

## Quick Start

### Installation

```bash
# Install with Logfire support
pip install -e ".[dev]"
```

### Run Full Evaluation

```bash
# Run on all test cases (400+ cases across 4 tools)
python evals/runner.py --output=report.json

# Run subset for quick check (40 cases)
python evals/runner.py --subset=10 --output=quick_report.json
```

### View Results

```bash
# Summary printed to console
cat report.json | jq '.overall_accuracy, .mcts_efficiency, .all_metrics_passed'
```

---

## Package Structure

```
evals/
├── __init__.py                 # Package exports
├── models.py                   # Data models (Case, ExpectedOutput, etc.)
├── metrics.py                  # All metric functions
├── dataset_generator.py        # Test case generation
├── runner.py                   # Main evaluation runner
├── compliance.py               # Audit export and security
├── linter.py                   # LLM-as-judge detector
├── ci_checker.py               # CI threshold checker
├── baseline_updater.py         # Baseline metrics updater
├── METRICS.md                  # Complete metrics catalog
├── ADDING_METRICS.md           # Guide for adding metrics
├── README.md                   # This file
├── baselines/
│   └── main_branch_metrics.json  # Baseline for regression comparison
└── datasets/
    └── (generated at runtime)
```

---

## Metrics Implemented

### Tool 1: Filter Transactions

- `filter_accuracy` - Binary threshold accuracy (>98%)
- `conversion_precision` - Currency conversion accuracy (>95%)
- `filter_iteration_efficiency` - MCTS budget adherence (100%)

### Tool 2: Classify Transactions

- `classification_accuracy` - Category accuracy (>90%)
- `confidence_calibration` - Confidence vs accuracy alignment (<10% error)
- `path_diversity` - MCTS exploration diversity (>60%)

### Tool 3: Detect Fraud

- `fraud_risk_accuracy` - Risk level accuracy (>92%)
- `critical_classification_strictness` - Zero false positives for CRITICAL (100%)
- `fraud_reward_convergence` - MCTS convergence rate (>95%)
- `fraud_indicator_coverage` - Indicator recall (>85%)

### Tool 4: Generate CSV

- `csv_column_completeness` - Required columns present (100%)
- `csv_row_count_accuracy` - Correct row count (100%)
- `explanation_validity` - Explanation keyword coverage (>80%)

### Composite Metrics

- `overall_agent_accuracy` - Macro-average across all tools (>90%)
- `mcts_efficiency_score` - MCTS computational efficiency (>75%)
- `cost_per_transaction` - Cost optimization (<$0.01)
- `fraud_false_positive_rate` - Critical fraud FPR (<2%)

See [METRICS.md](./METRICS.md) for complete catalog.

---

## CI/CD Integration

### Pre-commit Hook (REQ-EVAL-028)

Runs quick evaluation (10 cases, <60s) before each commit:

```bash
# Install
pre-commit install

# Run manually
pre-commit run quick-eval
```

**Blocks commit if**:
- Overall accuracy <85%
- MCTS convergence rate <90%

### Pull Request Check (REQ-EVAL-029)

Full evaluation (100 cases per tool) runs on every PR.

**Blocks merge if**:
- Overall accuracy drops >2% from baseline
- Any tool accuracy drops >5%
- Cost per transaction increases >10%
- MCTS efficiency drops >5%

See [.github/workflows/evaluation-ci.yml](../.github/workflows/evaluation-ci.yml)

---

## Baseline Comparison

### How It Works

1. Main branch has baseline metrics in `evals/baselines/main_branch_metrics.json`
2. PR evaluations compare against this baseline
3. CI fails if metrics regress beyond thresholds
4. When PR merges to main, baseline auto-updates

### Updating Baseline Manually

```bash
python evals/runner.py --no-baseline --output=report.json
python evals/baseline_updater.py report.json
git add evals/baselines/main_branch_metrics.json
git commit -m "chore: Update evaluation baseline"
```

---

## Logfire Integration

### Viewing Metrics

All metrics are logged to Logfire with structured data:

```
metric_name="fraud_risk_accuracy"
AND value >= 0.92
AND passes_threshold=true
```

### Dashboards

Logfire dashboards show:
- Time series of overall accuracy per commit
- Bar charts of per-tool accuracy
- Scatter plot of efficiency vs cost
- Heatmap of fraud FPR by category

### Alerts (REQ-LOGFIRE-METRIC-003)

Configured alerts:
- `fraud_false_positive_rate > 0.02`
- `mcts_efficiency_score < 0.70`
- `cost_per_transaction > 0.01`
- `convergence_failure_rate > 0.10`

---

## Compliance & Security

### Metric Immutability (REQ-METRIC-SEC-001)

All result models are frozen:

```python
class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)  # Cannot be modified
```

### Audit Export (REQ-METRIC-SEC-002)

Generate compliance export (7-year retention):

```python
from evals.compliance import export_metrics_for_compliance
from evals.models import EvaluationReport

# Load report
with open('report.json') as f:
    report = EvaluationReport.model_validate_json(f.read())

# Export for compliance
export_path = export_metrics_for_compliance(report, "audit_2025.csv")
```

Output CSV contains:
- `metric_name`
- `value`
- `calculated_at`
- `dataset_version`
- `evaluator_version`
- `git_commit`

---

## Dataset

### Generation

100 test cases per tool (400 total) with 20% adversarial examples:

```python
from evals.dataset_generator import generate_full_dataset

dataset = generate_full_dataset()
# {
#   "filter": [Case, Case, ...],      # 100 cases
#   "classify": [Case, Case, ...],    # 100 cases
#   "fraud": [Case, Case, ...],       # 100 cases
#   "csv": [Case, Case, ...],         # 100 cases
# }
```

### Adversarial Examples

- **Filter**: Amounts near 250 GBP threshold (249.99, 250.01)
- **Classify**: Ambiguous categories (business travel vs personal)
- **Fraud**: Synthetic fraud patterns
- **CSV**: Special characters, edge cases

---

## Adding New Metrics

See [ADDING_METRICS.md](./ADDING_METRICS.md) for complete guide.

**Quick steps**:
1. Add function to `evals/metrics.py`
2. Add to evaluator in `evals/runner.py`
3. Update thresholds in `evals/models.py`
4. Document in `evals/METRICS.md`
5. Run linter: `python evals/linter.py evals/metrics.py`

---

## Development

### Run Linter

Ensure no LLM-as-judge violations:

```bash
python evals/linter.py evals/*.py
```

### Run Tests

```bash
pytest tests/test_evals.py -v
```

### Generate New Dataset

```bash
python -c "
from evals.dataset_generator import generate_full_dataset
import json

dataset = generate_full_dataset()
print(f'Generated {sum(len(cases) for cases in dataset.values())} cases')
"
```

---

## Troubleshooting

### "No module named 'pydantic_evals'"

Install dependencies:
```bash
pip install -e ".[dev]"
```

### "Baseline file not found"

First run on branch - create baseline:
```bash
python evals/runner.py --no-baseline --output=report.json
python evals/baseline_updater.py report.json
```

### "LLM-as-judge violation detected"

Remove any LLM calls from metric functions. Metrics must be pure Python.

### Evaluation runs too slow

Use subset mode:
```bash
python evals/runner.py --subset=10  # 10 cases per tool
```

---

## Requirements Traceability

| Requirement | Implementation | Verification |
|-------------|----------------|--------------|
| REQ-EVAL-001 | `pyproject.toml` dependency | `pip list \| grep pydantic-evals` |
| REQ-EVAL-002 | `runner.py` random seed | Check `random_seed=42` in reports |
| REQ-EVAL-003 | `linter.py` enforcement | `python evals/linter.py` |
| REQ-EVAL-004 | `dataset_generator.py` | Run `generate_full_dataset()` |
| REQ-EVAL-005 | `models.py` MCTS metadata | Check `expected_min_iterations` |
| REQ-METRIC-001-013 | `metrics.py` individual | See [METRICS.md](./METRICS.md) |
| REQ-METRIC-014-017 | `metrics.py` composite | See [METRICS.md](./METRICS.md) |
| REQ-EVAL-028 | `.pre-commit-config.yaml` | `pre-commit run` |
| REQ-EVAL-029 | `.github/workflows/` | Check PR CI status |
| REQ-EVAL-030 | `baseline_updater.py` | Check `baselines/` directory |
| REQ-LOGFIRE-METRIC-001 | `runner.py` logging | Query Logfire |
| REQ-LOGFIRE-METRIC-002 | Logfire dashboards | Check Logfire console |
| REQ-LOGFIRE-METRIC-003 | Logfire alerts | Check alert config |
| REQ-METRIC-SEC-001 | `models.py` frozen | Try modifying report |
| REQ-METRIC-SEC-002 | `compliance.py` export | Run `export_metrics_for_compliance()` |

---

## Version

**Version**: 1.0.0
**Last Updated**: 2025-01-09
**Requirements Spec**: Pure Mathematical Evals & Metrics Requirements Specification

---

## License

Same as parent project.
