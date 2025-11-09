# Guide: Adding New Metrics

**REQ-DOC-002: Step-by-step guide for adding evaluation metrics**

This guide walks you through adding a new metric to the evaluation framework.

---

## Overview

Adding a new metric requires:
1. ✅ Defining the metric function (pure Python, no LLM)
2. ✅ Adding it to the evaluator
3. ✅ Updating documentation
4. ✅ Adding Logfire logging
5. ✅ Updating CI thresholds

---

## Step 1: Create the Metric Function

**Location**: `evals/metrics.py`

### Template

```python
def my_new_metric(output: ToolResult, expected: ExpectedOutput) -> float:
    """
    Brief description of what this metric measures.

    Formula: Describe the mathematical formula
    Target: Describe the target threshold

    Args:
        output: Result from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Score (typically 0.0 to 1.0)
    """
    # Your pure Python calculation here
    # Example:
    if output.some_field == expected.some_expected_field:
        return 1.0
    else:
        return 0.0
```

### Rules

**CRITICAL**: Your metric MUST be a pure Python function. No LLM calls allowed!

✅ **Allowed**:
- Arithmetic operations (`+`, `-`, `*`, `/`)
- Comparisons (`==`, `!=`, `<`, `>`)
- String operations (`in`, `.lower()`, `.count()`)
- Set operations (`&`, `-`, `|`)
- NumPy functions (mean, std, etc.)
- Statistical formulas (precision, recall, F1, etc.)

❌ **Forbidden**:
- Any LLM API calls
- `LLMJudge`, `GEval`, etc.
- `llm.generate()`, `ai.score()`, etc.

### Example: Precision Metric

```python
def fraud_precision(output: FraudResult, expected: ExpectedOutput) -> float:
    """
    Precision for fraud indicator detection.

    Formula: TP / (TP + FP)
    Target: >0.85

    Args:
        output: FraudResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Precision score (0.0 to 1.0)
    """
    expected_indicators = set(expected.expected_fraud_indicators)
    found_indicators = set(output.fraud_indicators)

    true_positives = len(expected_indicators & found_indicators)
    false_positives = len(found_indicators - expected_indicators)

    total_predicted = true_positives + false_positives
    if total_predicted == 0:
        return 0.0

    return true_positives / total_predicted
```

---

## Step 2: Add to Evaluator

**Location**: `evals/runner.py`

### Add Import

```python
from evals.metrics import (
    # ... existing metrics ...
    my_new_metric,  # Add your metric here
)
```

### Add to Tool Evaluation Method

Find the appropriate `_evaluate_*_tool` method and add your metric:

```python
def _evaluate_fraud_tool(self, cases: List[Case]) -> ToolEvaluationResult:
    with logfire.span("evaluate_fraud_tool", total_cases=len(cases)):
        metrics_sum = {
            # ... existing metrics ...
            "my_new_metric": 0.0,  # Add here
        }

        for case in cases:
            # ... existing code ...

            # Calculate your metric
            new_metric_value = my_new_metric(output, case.expected_output)
            metrics_sum["my_new_metric"] += new_metric_value

            # Log to Logfire
            self._log_metric("my_new_metric", new_metric_value, 0.85)  # threshold

        # ... rest of method ...
```

---

## Step 3: Update Thresholds

**Location**: `evals/models.py`

Add your metric threshold to `MetricThresholds`:

```python
class MetricThresholds(BaseModel):
    """Thresholds for all metrics."""

    # ... existing thresholds ...

    my_new_metric: float = Field(0.85, description="REQ-METRIC-XXX: >85%")
```

---

## Step 4: Add Logfire Logging

Your metric is automatically logged if you added `self._log_metric()` in Step 2.

### Optional: Add Dashboard Widget

To add a dashboard widget in Logfire:

1. Log into Logfire console
2. Navigate to Dashboards
3. Add new widget with query:
   ```
   metric_name="my_new_metric"
   ```
4. Choose visualization (line chart, bar chart, etc.)

---

## Step 5: Update Documentation

**Location**: `evals/METRICS.md`

Add your metric to the appropriate section:

```markdown
### my_new_metric

**REQ-METRIC-XXX**

- **Name**: `my_new_metric`
- **Formula**: `TP / (TP + FP)`
- **Threshold**: >0.85
- **Purpose**: Measures precision of fraud indicator detection
- **Returns**: Precision score (0.0 to 1.0)
- **Logfire Query**: `metric_name="my_new_metric"`
```

---

## Step 6: Update CI Thresholds (Optional)

If your metric should block PRs, update `evals/ci_checker.py`:

```python
def check_thresholds(report_path: str) -> bool:
    # ... existing code ...

    # Check my new metric
    my_metric_value = report.get("my_new_metric", 0.0)
    if my_metric_value < 0.85:
        checks.append((False, f"my_new_metric below threshold: {my_metric_value:.2%}"))
    else:
        checks.append((True, f"my_new_metric: {my_metric_value:.2%}"))

    # ... rest of function ...
```

---

## Step 7: Update Tests (Recommended)

**Location**: `tests/test_evals.py` (create if needed)

Add a unit test for your metric:

```python
def test_my_new_metric():
    """Test my_new_metric calculation."""
    # Create mock output
    output = FraudResult(
        transaction_id="TEST001",
        risk_level=FraudRiskLevel.HIGH,
        confidence=0.9,
        fraud_indicators=["high_amount", "suspicious_merchant", "false_positive"],
        mcts_path=["action1"],
        mcts_reward=0.75,
        mcts_metadata=create_mock_metadata(),
    )

    # Create expected output
    expected = ExpectedOutput(
        tool_1_filtered=True,
        tool_2_classification="Personal",
        tool_3_fraud_risk="HIGH",
        tool_3_confidence=0.9,
        expected_fraud_indicators=["high_amount", "suspicious_merchant"],
        tool_4_columns_complete=True,
    )

    # Calculate metric
    precision = fraud_precision(output, expected)

    # Expected: 2 TP, 1 FP → 2/3 = 0.667
    assert abs(precision - 0.667) < 0.01
```

---

## Step 8: Update Expected Outputs

**Location**: `evals/dataset_generator.py`

If your metric requires new expected values, update `ExpectedOutput` model:

```python
class ExpectedOutput(BaseModel):
    # ... existing fields ...

    expected_my_new_field: Optional[float] = Field(None, description="Expected value for my metric")
```

Then update the dataset generator to populate this field.

---

## Step 9: Verify No LLM-as-Judge

Run the linter to ensure your metric doesn't violate REQ-EVAL-003:

```bash
python evals/linter.py evals/metrics.py
```

Should output:
```
✅ No LLM-as-judge violations found
```

---

## Step 10: Test End-to-End

Run a quick evaluation to ensure everything works:

```bash
python evals/runner.py --subset=10
```

Check the output for your new metric.

---

## Common Patterns

### Binary Classification Metric

```python
def my_binary_metric(output, expected) -> float:
    return 1.0 if output.field == expected.expected_field else 0.0
```

### Continuous Score with Error

```python
def my_continuous_metric(output, expected) -> float:
    error = abs(output.value - expected.expected_value)
    max_error = expected.max_acceptable_error
    return max(0.0, 1.0 - (error / max_error))
```

### Recall (Coverage)

```python
def my_recall_metric(output, expected) -> float:
    expected_items = set(expected.expected_items)
    found_items = set(output.found_items)

    if len(expected_items) == 0:
        return 1.0

    true_positives = len(expected_items & found_items)
    false_negatives = len(expected_items - found_items)

    return true_positives / (true_positives + false_negatives)
```

### Precision

```python
def my_precision_metric(output, expected) -> float:
    expected_items = set(expected.expected_items)
    found_items = set(output.found_items)

    if len(found_items) == 0:
        return 0.0

    true_positives = len(expected_items & found_items)
    false_positives = len(found_items - expected_items)

    return true_positives / (true_positives + false_positives)
```

### F1 Score

```python
def my_f1_metric(output, expected) -> float:
    precision = my_precision_metric(output, expected)
    recall = my_recall_metric(output, expected)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
```

---

## Checklist

Before submitting your metric:

- [ ] Metric function is pure Python (no LLM calls)
- [ ] Function has docstring with formula and target
- [ ] Added to `evals/metrics.py`
- [ ] Added to evaluator in `evals/runner.py`
- [ ] Threshold defined in `MetricThresholds`
- [ ] Logfire logging added
- [ ] Documented in `evals/METRICS.md`
- [ ] Linter passes (`python evals/linter.py evals/metrics.py`)
- [ ] Test added (recommended)
- [ ] Quick evaluation runs successfully

---

## Need Help?

- **Question about formula**: Check existing metrics in `evals/metrics.py` for examples
- **Logfire issues**: See Logfire documentation at https://logfire.pydantic.dev/
- **General evaluation questions**: Review the requirements in the original spec

---

**Remember**: Every metric must be a deterministic Python function. If you find yourself wanting to call an LLM to score something, step back and ask: "What mathematical property am I actually trying to measure?"

---

**Last Updated**: 2025-01-09
**Version**: 1.0.0
