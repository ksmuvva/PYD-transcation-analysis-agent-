# Test Bugs and Issues Report

## Test Execution Date
2025-11-09

## Summary
This document tracks all bugs, issues, and failures discovered during comprehensive testing of the transaction analysis agent.

---

## CRITICAL BUGS

### BUG-001: AnthropicModel initialization error
**Severity:** CRITICAL
**Component:** src/config.py:158
**Test:** tests/test_tools.py (11 errors)

**Description:**
`AnthropicModel.__init__()` is being called with `api_key` parameter, but the current pydantic-ai API doesn't accept this parameter directly.

**Error Message:**
```
TypeError: AnthropicModel.__init__() got an unexpected keyword argument 'api_key'
```

**Root Cause:**
The pydantic-ai library has changed its API. `AnthropicModel` now reads the API key from environment variables (ANTHROPIC_API_KEY) or requires a custom provider with AsyncAnthropic client.

**Impact:**
- All tests using Anthropic provider fail
- Agent cannot be initialized with Anthropic models
- Filter tool tests fail (11 errors)

**Current Signature:**
```python
AnthropicModel(model_name, *, provider='anthropic', profile=None, settings=None)
```

**Fix Required:**
```python
# Option 1: Use environment variable (RECOMMENDED)
os.environ['ANTHROPIC_API_KEY'] = config.api_key
return AnthropicModel(config.model)

# Option 2: Use custom provider
from anthropic import AsyncAnthropic
client = AsyncAnthropic(api_key=config.api_key)
# Then use custom provider approach
```

**Affected Tests:**
- test_filter_with_default_threshold
- test_filter_with_custom_threshold
- test_filter_with_high_threshold
- test_filter_with_zero_threshold
- test_filter_currency_conversion_usd_to_gbp
- test_filter_currency_conversion_eur_to_gbp
- test_filter_mixed_currencies
- test_filter_custom_currency
- test_filter_results_stored_in_context
- test_filter_empty_dataframe
- test_filter_negative_threshold_rejected

---

### BUG-002: Incorrect import path for convert_to_gbp
**Severity:** HIGH
**Component:** tests/test_tools.py:280,287,295,303,311,318
**Test:** TestCurrencyConversion (6 failures)

**Description:**
Tests are trying to import `convert_to_gbp` as a module-level function, but it's actually a static method of the `CSVProcessor` class.

**Error Message:**
```
ImportError: cannot import name 'convert_to_gbp' from 'src.csv_processor'
```

**Root Cause:**
Test code uses:
```python
from src.csv_processor import convert_to_gbp
```

But the actual implementation is:
```python
class CSVProcessor:
    @staticmethod
    def convert_to_gbp(amount: float, currency: Currency) -> float:
        ...
```

**Impact:**
- 6 currency conversion tests fail
- Cannot verify currency conversion logic in isolation

**Fix Required:**
```python
# Change test imports from:
from src.csv_processor import convert_to_gbp

# To:
from src.csv_processor import CSVProcessor
# Then use:
CSVProcessor.convert_to_gbp(...)
```

**Affected Tests:**
- test_gbp_to_gbp_conversion
- test_usd_to_gbp_conversion
- test_eur_to_gbp_conversion
- test_jpy_to_gbp_conversion
- test_zero_amount_conversion
- test_large_amount_conversion

---

## TEST RESULTS SUMMARY

### tests/test_models.py
**Status:** ✅ PASSED
**Results:** 6/6 passed
**Details:**
- All Pydantic model validations work correctly
- Transaction model validation works
- Date parsing from strings works
- Confidence score bounds validation works
- Enum validation works

### tests/test_tools.py
**Status:** ❌ FAILED
**Results:** 2/19 passed, 11 errors, 6 failures
**Pass Rate:** 10.5%
**Details:**
- Basic tool existence checks pass (2 tests)
- All filter tool tests error due to BUG-001 (11 errors)
- All currency conversion tests fail due to BUG-002 (6 failures)

---

### BUG-003: MCTSEngine initialization uses wrong parameter name
**Severity:** HIGH
**Component:** tests/test_tools.py:79
**Test:** TestFilterTransactionsTool (11 errors)
**Status:** ✅ FIXED

**Description:**
Test fixture `real_mcts_engine` was trying to initialize `MCTSEngine` with `llm_client` parameter, but the actual constructor expects `llm_function` (a callable).

**Fix Applied:**
Created wrapper function that converts LLM client to callable function.

---

### BUG-004: Tool functions expect RunContext but tests call them directly
**Severity:** HIGH
**Component:** tests/test_tools.py - All TestFilterTransactionsTool tests
**Status:** ✅ FIXED

**Description:**
Tool functions are decorated with `@financial_agent.tool` and expect `RunContext[AgentDependencies]`, but tests were calling them directly with `AgentDependencies`.

**Fix Applied:**
Created `mock_ctx` fixture that wraps AgentDependencies in a MockRunContext class.

---

### BUG-005: Test code has stale references to agent_deps
**Severity:** MEDIUM
**Component:** tests/test_tools.py - Multiple test methods
**Status:** PARTIALLY FIXED

**Description:**
After fixing BUG-004, some tests still have references to `agent_deps` variable instead of using `mock_ctx.deps`.

**Impact:**
- 7/11 FilterTransactionsTool tests still fail
- Tests reference undefined variable `agent_deps`

**Remaining Issues:**
1. test_filter_currency_conversion_usd_to_gbp: Line 182 uses `agent_deps.config`
2. test_filter_currency_conversion_eur_to_gbp: Line 201 uses `agent_deps.config`
3. test_filter_custom_currency: Line 218-221 uses `agent_deps` directly
4. test_filter_results_stored_in_context: Uses `agent_deps`
5. test_filter_empty_dataframe: Pandas bug with empty dataframes
6. test_filter_with_zero_threshold: Assertion error (expected 5, got 3)
7. test_filter_negative_threshold_rejected: Incorrect pytest.raises syntax

---

### BUG-006: MCTS test methods call ucb1_score incorrectly
**Severity:** MEDIUM
**Component:** tests/test_mcts.py - Multiple tests
**Status:** NOT FIXED

**Description:**
Many MCTS tests fail because they call `ucb1_score()` without required `exploration_constant` parameter, or with wrong keyword argument `c`.

**Impact:**
- 14/39 MCTS tests fail
- Cannot verify UCB1 algorithm correctness
- Cannot verify tree traversal logic

**Error Examples:**
```
TypeError: MCTSNode.ucb1_score() missing 1 required positional argument: 'exploration_constant'
TypeError: MCTSNode.ucb1_score() got an unexpected keyword argument 'c'
```

---

### BUG-007: CSVProcessor.add_gbp_column fails on empty DataFrames
**Severity:** LOW
**Component:** src/csv_processor.py:191
**Test:** test_filter_empty_dataframe

**Description:**
When applying conversion to empty DataFrame, pandas raises ValueError about setting multiple columns to single column.

**Error:**
```
ValueError: Cannot set a DataFrame with multiple columns to the single column amount_gbp
```

**Fix Required:**
Add check for empty DataFrame before applying conversion.

---

## BUGS FIXED

### BUG-001: AnthropicModel initialization error
**Status:** ✅ FIXED
**Fix Applied:** Modified src/config.py:158 to set API key in environment variable before creating AnthropicModel

### BUG-002: Incorrect import path for convert_to_gbp
**Status:** ✅ FIXED
**Fix Applied:** Updated tests/test_tools.py to import CSVProcessor class and use CSVProcessor.convert_to_gbp()

### BUG-003: MCTSEngine initialization uses wrong parameter name
**Status:** ✅ FIXED
**Fix Applied:** Created wrapper function that converts LLM client to callable function

### BUG-004: Tool functions expect RunContext but tests call them directly
**Status:** ✅ FIXED
**Fix Applied:** Created mock_ctx fixtures that wrap AgentDependencies in MockRunContext class

### BUG-005: Test code has stale references to agent_deps
**Status:** ✅ FIXED
**Fix Applied:**
- Added mock_ctx fixture to test_tool_interactions.py
- Replaced all agent_deps references with mock_ctx
- Fixed recursive fixture dependency
- Updated result key assertions from 'filtered_transactions' to 'filtered_df'

### BUG-007: CSVProcessor.add_gbp_column fails on empty DataFrames
**Status:** ✅ FIXED
**Fix Applied:** Added check for empty DataFrame before applying conversion in CSVProcessor.add_gbp_column()

### BUG-008: Zero threshold parameter handling (NEW)
**Status:** ✅ FIXED
**Component:** src/agent.py:110
**Description:** When threshold=0.0 is explicitly passed, it was treated as falsy
**Fix Applied:** Changed to `threshold if threshold is not None else config.threshold_amount`

### BUG-009: None/NaN transaction_id validation (NEW)
**Status:** ✅ FIXED
**Component:** src/csv_processor.py
**Description:** None/NaN transaction_ids were being converted to string "None"
**Fix Applied:** Added validation to reject None/NaN transaction_ids

---

## LATEST TEST EXECUTION (2025-11-09)

**Test Results:**
- ✅ 164 tests PASSING (94% pass rate excluding CLI tests)
- ❌ 8 tests FAILING (tool interaction tests needing transaction_data parameter)
- ⏭️ 9 tests SKIPPED (integration tests requiring API)

**Progress:** Fixed 33 test failures from previous run

---

## TEST EXECUTION SUMMARY

### ✅ Successfully Tested
1. **tests/test_models.py** - 6/6 tests PASSED (100%)
   - Transaction model validation
   - Date parsing
   - Confidence score bounds
   - Enum validation

2. **tests/test_tools.py::TestCurrencyConversion** - 6/6 tests PASSED (100%)
   - GBP to GBP conversion
   - USD to GBP conversion
   - EUR to GBP conversion
   - JPY to GBP conversion
   - Zero and large amount conversions

3. **tests/test_tools.py::TestFilterTransactionsTool** - 4/11 tests PASSED (36%)
   - Default threshold filtering
   - Custom threshold filtering
   - High threshold filtering
   - Mixed currency filtering

4. **tests/test_mcts.py** - 25/39 tests PASSED (64%)
   - Engine initialization
   - Hypothesis generation (classification & fraud)
   - Hypothesis evaluation
   - Backpropagation logic
   - Search convergence
   - Edge cases (empty tree, deep tree, etc.)

### ❌ Tests with Failures
1. **tests/test_tools.py::TestFilterTransactionsTool** - 7/11 FAILED
   - Mostly due to stale test code (BUG-005)
   - Empty dataframe handling (BUG-007)

2. **tests/test_mcts.py** - 14/39 FAILED
   - UCB1 score method signature mismatches (BUG-006)
   - Node relationship assertions

### ⏸️ Tests Not Run
- tests/test_prompts.py
- tests/test_tool_interactions.py
- tests/test_edge_cases.py
- tests/test_e2e.py
- tests/test_cli.py
- tests/test_integration.py (requires live API calls)

---

## OVERALL TEST METRICS

**Total Tests Run:** 82
**Tests Passed:** 41 (50%)
**Tests Failed:** 41 (50%)

**Critical Bugs Found:** 3 (all fixed)
**High Severity Bugs:** 2 (1 fixed, 1 partially fixed)
**Medium Severity Bugs:** 2 (0 fixed)
**Low Severity Bugs:** 1 (0 fixed)

---

## SYNTHETIC DATASETS

**Generated:** 25 datasets with varying characteristics
**Categories:**
- Small balanced (5 transactions each) - 5 datasets
- Medium varied (20 transactions each) - 5 datasets
- Large realistic (50 transactions each) - 5 datasets
- Edge cases (various) - 10 datasets

**Ground Truth:** Available for all 25 datasets with:
- Transaction categorization labels
- Fraud risk level labels
- GBP amount calculations
- Above/below threshold indicators

**Total Synthetic Transactions:** 545

---

## RECOMMENDATIONS

### Immediate Priority (Critical for Production)
1. ✅ Fix BUG-001: AnthropicModel API key handling - **FIXED**
2. ✅ Fix BUG-002: Currency conversion imports - **FIXED**
3. ✅ Fix BUG-003: MCTS Engine initialization - **FIXED**
4. ✅ Fix BUG-004: Tool RunContext requirements - **FIXED**
5. ⚠️ Fix BUG-005: Complete test code cleanup for agent_deps references
6. ⚠️ Fix BUG-007: Empty DataFrame handling in CSV processor

### Medium Priority (Testing Improvements)
1. Fix BUG-006: MCTS test method signatures
2. Complete test_tools.py remaining failures
3. Run full integration tests with real API
4. Test all 25 synthetic datasets end-to-end

### Low Priority (Nice to Have)
1. Add performance benchmarks
2. Add load testing (1000+ transactions)
3. Test concurrent operations
4. Enhanced telemetry validation

---

## LOGFIRE TELEMETRY STATUS

**Configuration:** Enabled in .env
**Project:** starter-project
**Expected URL:** https://logfire-eu.pydantic.dev/ksmuvva/starter-project

**Note:** Telemetry integration is configured but not yet verified with live API calls. Integration tests are needed to confirm traces appear in Logfire.

---

## NEXT STEPS FOR COMPLETION

1. Run CLI-based test on one synthetic dataset to verify end-to-end functionality
2. Fix remaining test failures in test_tools.py
3. Execute comprehensive dataset testing (all 25 datasets)
4. Verify Logfire telemetry with actual traces
5. Calculate comprehensive accuracy metrics
6. Generate final test report with efficacy analysis

---

## FILES CREATED

1. `/tests/generate_synthetic_data.py` - Dataset generator script
2. `/tests/synthetic_datasets/` - 25 CSV datasets + 25 ground truth JSON files
3. `/test_all_datasets.py` - Comprehensive dataset testing script
4. `/run_comprehensive_tests.py` - Test suite runner
5. `/bugs_test.md` - This comprehensive bug report (THIS FILE)
6. `/.env` - Environment configuration with Anthropic API key

---

## CODE FIXES APPLIED

1. **src/config.py:158** - Set API key in environment for AnthropicModel
2. **tests/test_tools.py:75-92** - Created LLM wrapper function for MCTS
3. **tests/test_tools.py:113-120** - Created mock_ctx fixture
4. **tests/test_tools.py:278-321** - Fixed currency conversion test imports
5. **tests/test_tools.py** - Bulk replacement of agent_deps with mock_ctx

---

## NOTES

- Python 3.11.14 with uv package manager
- All dependencies successfully installed
- Anthropic API key properly configured
- Virtual environment activated
- Pydantic Logfire configured
- 25 synthetic datasets generated with ground truth
- Critical bugs blocking testing have been fixed
- System is ready for comprehensive end-to-end testing
