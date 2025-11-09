# Metrics Catalog

**REQ-DOC-001: Complete catalog of all evaluation metrics**

This document provides detailed information about every metric used in the transaction analysis agent evaluation framework.

---

## Table of Contents

- [Tool 1: Filter Metrics](#tool-1-filter-metrics)
- [Tool 2: Classification Metrics](#tool-2-classification-metrics)
- [Tool 3: Fraud Detection Metrics](#tool-3-fraud-detection-metrics)
- [Tool 4: CSV Generation Metrics](#tool-4-csv-generation-metrics)
- [Composite Metrics](#composite-metrics)
- [Logfire Queries](#logfire-queries)

---

## Tool 1: Filter Metrics

### filter_accuracy

**REQ-METRIC-001**

- **Name**: `filter_accuracy`
- **Formula**: `1.0 if is_above_threshold == expected.tool_1_filtered else 0.0`
- **Threshold**: >98% (0.98)
- **Purpose**: Ensures binary filter decision (≥250 GBP) matches ground truth
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="filter_accuracy" AND environment="production"`

### conversion_precision

**REQ-METRIC-002**

- **Name**: `conversion_precision`
- **Formula**: `max(0.0, 1.0 - (|actual_gbp - expected_gbp| / expected_gbp))`
- **Threshold**: >0.95 (mean precision)
- **Purpose**: Measures accuracy of currency conversion to GBP
- **Returns**: Precision score (0.0 to 1.0)
- **Logfire Query**: `metric_name="conversion_precision"`

### filter_iteration_efficiency

**REQ-METRIC-003**

- **Name**: `filter_iteration_efficiency`
- **Formula**: `1.0 if iterations_used <= budget else 0.0`
- **Threshold**: 1.0 (100% efficiency)
- **Purpose**: Ensures MCTS doesn't exceed iteration budget (100 for filter)
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="filter_iteration_efficiency"`

---

## Tool 2: Classification Metrics

### classification_accuracy

**REQ-METRIC-004**

- **Name**: `classification_accuracy`
- **Formula**: `1.0 if category == expected.tool_2_classification else 0.0`
- **Threshold**: >90% (0.90)
- **Purpose**: Exact match on transaction category (Business/Personal/Investment/Gambling)
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="classification_accuracy"`

### confidence_calibration

**REQ-METRIC-005**

- **Name**: `confidence_calibration`
- **Formula**: `max(0.0, 1.0 - |confidence - accuracy|)`
- **Threshold**: >0.90 (calibration error <0.10)
- **Purpose**: Measures if confidence scores match actual accuracy (well-calibrated model)
- **Returns**: Calibration score (0.0 to 1.0)
- **Logfire Query**: `metric_name="confidence_calibration"`

### path_diversity

**REQ-METRIC-006**

- **Name**: `path_diversity`
- **Formula**: `min(unique_actions / total_possible_actions, 1.0)`
- **Threshold**: >0.60 (explores >60% of action space)
- **Purpose**: Rewards MCTS for exploring diverse actions, not getting stuck in local optima
- **Returns**: Diversity ratio (0.0 to 1.0)
- **Logfire Query**: `metric_name="path_diversity"`

---

## Tool 3: Fraud Detection Metrics

### fraud_risk_accuracy

**REQ-METRIC-007**

- **Name**: `fraud_risk_accuracy`
- **Formula**: `1.0 if risk_level == expected.tool_3_fraud_risk else 0.0`
- **Threshold**: >92% (0.92)
- **Purpose**: Exact match on risk level (LOW/MEDIUM/HIGH/CRITICAL)
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="fraud_risk_accuracy"`

### critical_classification_strictness

**REQ-METRIC-008**

- **Name**: `critical_classification_strictness`
- **Formula**:
  - If expected CRITICAL: `1.0 if predicted CRITICAL else 0.0`
  - If expected not CRITICAL: `1.0 if not predicted CRITICAL else 0.5`
- **Threshold**: 1.0 (zero false positives for CRITICAL)
- **Purpose**: Safety-critical - must not falsely flag CRITICAL, must catch all true CRITICAL
- **Returns**: Strictness score (0.0, 0.5, or 1.0)
- **Logfire Query**: `metric_name="critical_classification_strictness"`

### fraud_reward_convergence

**REQ-METRIC-009**

- **Name**: `fraud_reward_convergence`
- **Formula**: `1.0 if final_variance <= 0.05 else 0.0`
- **Threshold**: >95% (convergence rate)
- **Purpose**: Ensures MCTS reward variance is low (converged to stable solution)
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="fraud_reward_convergence"`

### fraud_indicator_coverage

**REQ-METRIC-010**

- **Name**: `fraud_indicator_coverage`
- **Formula**: `TP / (TP + FN)` where TP = found expected indicators, FN = missed indicators
- **Threshold**: >0.85 (finds 85%+ of true indicators)
- **Purpose**: Measures recall - did MCTS find all the fraud signals?
- **Returns**: Coverage ratio (0.0 to 1.0)
- **Logfire Query**: `metric_name="fraud_indicator_coverage"`

---

## Tool 4: CSV Generation Metrics

### csv_column_completeness

**REQ-METRIC-011**

- **Name**: `csv_column_completeness`
- **Formula**: `present_columns / required_columns` (4 required: classification, fraud_risk, confidence, mcts_explanation)
- **Threshold**: 1.0 (100% completeness)
- **Purpose**: Ensures all required output columns are present in CSV
- **Returns**: Completeness ratio (0.0 to 1.0, increments of 0.25)
- **Logfire Query**: `metric_name="csv_column_completeness"`

### csv_row_count_accuracy

**REQ-METRIC-012**

- **Name**: `csv_row_count_accuracy`
- **Formula**: `1.0 if row_count == expected_row_count else 0.0`
- **Threshold**: 1.0 (100% accuracy)
- **Purpose**: Ensures correct number of rows in output CSV
- **Returns**: Binary score (0.0 or 1.0)
- **Logfire Query**: `metric_name="csv_row_count_accuracy"`

### explanation_validity

**REQ-METRIC-013**

- **Name**: `explanation_validity`
- **Formula**: `keywords_found / keywords_required`
- **Threshold**: >0.80 (80% of keywords present)
- **Purpose**: Checks if MCTS explanations contain expected reasoning keywords
- **Returns**: Keyword coverage ratio (0.0 to 1.0)
- **Logfire Query**: `metric_name="explanation_validity"`

---

## Composite Metrics

### overall_agent_accuracy

**REQ-METRIC-014**

- **Name**: `overall_agent_accuracy`
- **Formula**: `mean([filter_accuracy, classification_accuracy, fraud_risk_accuracy, csv_column_completeness])`
- **Threshold**: >0.90 for production release
- **Purpose**: High-level health metric - macro-average across all tools
- **Returns**: Accuracy score (0.0 to 1.0)
- **Logfire Query**: `metric_name="overall_agent_accuracy"`

### mcts_efficiency_score

**REQ-METRIC-015**

- **Name**: `mcts_efficiency_score`
- **Formula**: `geometric_mean([filter_iteration_efficiency, fraud_reward_convergence, path_diversity])`
- **Threshold**: >0.75 (efficient and robust)
- **Purpose**: Measures MCTS computational efficiency and exploration quality. Uses geometric mean to penalize poor performance in any dimension.
- **Returns**: Efficiency score (0.0 to 1.0)
- **Logfire Query**: `metric_name="mcts_efficiency_score"`

### cost_per_transaction

**REQ-METRIC-016**

- **Name**: `cost_per_transaction`
- **Formula**: `total_cost / case_count`
- **Threshold**: <$0.01 for POC, <$0.005 for production
- **Purpose**: Rollup metric for cost optimization (lower is better)
- **Returns**: Cost in USD
- **Logfire Query**: `metric_name="cost_per_transaction"`

### fraud_false_positive_rate

**REQ-METRIC-017**

- **Name**: `fraud_false_positive_rate`
- **Formula**: `FP / (FP + TN)` for CRITICAL risk level
- **Threshold**: <2% (0.02) - critical for compliance
- **Purpose**: Measures false alarm rate for CRITICAL fraud flags (must be very low)
- **Returns**: False positive rate (0.0 to 1.0)
- **Logfire Query**: `metric_name="fraud_false_positive_rate"`

---

## Logfire Queries

### View All Metrics for a Run

```
run_id="<your-run-id>"
```

### View Failed Metrics

```
metric_name=* AND passes_threshold=false
```

### View Metrics by Tool

```
span_name="evaluate_filter_tool"
span_name="evaluate_classify_tool"
span_name="evaluate_fraud_tool"
span_name="evaluate_csv_tool"
```

### View Baseline Comparisons

```
span_name="baseline_comparison"
```

### Alert Conditions (REQ-LOGFIRE-METRIC-003)

The following Logfire alerts are configured:

- `fraud_false_positive_rate > 0.02` → Alert: "CRITICAL fraud FPR exceeded"
- `mcts_efficiency_score < 0.70` → Alert: "MCTS efficiency degraded"
- `cost_per_transaction > 0.01` → Alert: "Cost per transaction too high"
- `fraud_reward_convergence < 0.90` → Alert: "MCTS convergence failure rate high"

---

## Mathematical Properties

### No LLM Involvement (REQ-EVAL-003)

**Every metric is a deterministic Python function.** No metrics call an LLM to produce scores.

**Verification**: Run `python evals/linter.py evals/metrics.py` to verify no LLM-as-judge imports.

### Reproducibility (REQ-EVAL-002)

All evaluations run with `random_seed=42` by default, ensuring:
- Identical dataset generation
- Reproducible MCTS exploration (if MCTS uses seeded random)
- Consistent metric values across runs

### Immutability (REQ-METRIC-SEC-001)

All evaluation results are stored in frozen Pydantic models:
- `EvaluationReport` (frozen=True)
- `MetricResult` (frozen=True)
- `Case` (frozen=True)

Once calculated, metrics cannot be modified.

---

## Metric Relationships

```
overall_agent_accuracy
  ├─ filter_accuracy (Tool 1)
  ├─ classification_accuracy (Tool 2)
  ├─ fraud_risk_accuracy (Tool 3)
  └─ csv_column_completeness (Tool 4)

mcts_efficiency_score
  ├─ filter_iteration_efficiency
  ├─ fraud_reward_convergence
  └─ path_diversity
```

---

## Adding New Metrics

See [ADDING_METRICS.md](./ADDING_METRICS.md) for step-by-step guide.

---

## Compliance

- **REQ-EVAL-003**: ✅ No LLM-as-judge (all pure Python)
- **REQ-METRIC-SEC-001**: ✅ Immutable results (frozen models)
- **REQ-METRIC-SEC-002**: ✅ Audit export available (`evals/compliance.py`)
- **REQ-LOGFIRE-METRIC-001**: ✅ All metrics logged to Logfire
- **REQ-LOGFIRE-METRIC-003**: ✅ Alerts configured for regression

---

**Last Updated**: 2025-01-09
**Version**: 1.0.0
