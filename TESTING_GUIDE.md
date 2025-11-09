# Comprehensive Testing Guide

## Overview

This project includes exhaustive integration tests for the MCTS transaction analysis engine using **real OpenAI API calls** (no mocks). The tests cover all tools, multiple OpenAI reasoning models, synthetic data scenarios, and edge cases.

## Test Components

### 1. Synthetic Data Generator (`tests/synthetic_data_generator.py`)

Generates realistic financial transaction data for testing:

- **Normal transactions**: Business, Personal, Travel, Entertainment, Investment
- **Suspicious transactions**: High-risk patterns, unusual merchants
- **Fraud scenarios**: Rapid succession, unusual amounts, suspicious merchants
- **Edge cases**: Very large/small amounts, exact threshold values, multiple currencies
- **Configurable distributions**: Control category and fraud risk distributions

**Usage:**
```python
from tests.synthetic_data_generator import create_test_dataset, SyntheticTransactionGenerator

# Create test datasets
df_small = create_test_dataset("small", include_fraud=True)
df_large = create_test_dataset("large", include_fraud=True)

# Generate specific scenarios
generator = SyntheticTransactionGenerator(seed=42)
fraud_txns = generator.generate_fraud_scenario("rapid_succession")
edge_cases = generator.generate_edge_cases()
```

### 2. OpenAI Integration Tests (`tests/test_openai_mcts_integration.py`)

Comprehensive integration tests using real OpenAI API:

**Test Categories:**

1. **LLM Client Setup** - Validate OpenAI client creation and model configuration
2. **MCTS Engine** - Test core MCTS algorithm with real API
3. **Tool 1: Filter Transactions** - Test filtering with currency conversion
4. **Tool 2: Classify Transactions** - Test classification across categories
5. **Tool 3: Fraud Detection** - Test fraud detection across risk levels
6. **Tool 4: CSV Generation** - Test enhanced CSV output
7. **Model Comparison** - Compare o1-mini vs o1-preview vs o1
8. **Stress Tests** - Large amounts, small amounts, batch processing
9. **Error Handling** - Invalid data, API errors
10. **Performance Benchmarks** - Latency and throughput measurements

**Tested Models:**
- `o1-mini` (cost-effective, fast)
- `o1-preview` (more capable, higher cost)
- `o1` (most capable, highest cost) - optional
- `o3-mini` (if available)

## Running Tests

### Prerequisites

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

### Quick Start

#### 1. Fast Tests (Recommended for development)

```bash
# Run fast tests with o1-mini (most cost-effective)
python run_openai_tests.py --fast

# Or directly with pytest
pytest tests/test_openai_mcts_integration.py -v -m "not slow" --model=o1-mini
```

**Cost estimate:** ~$0.30-0.50 USD
**Time:** ~5-10 minutes

#### 2. Comprehensive Tests

```bash
# Run all tests with all models (WARNING: expensive!)
python run_openai_tests.py --all

# Run all tests with specific model
python run_openai_tests.py --all --model o1-mini
```

**Cost estimate:** $1.50-10.00 USD (depending on model)
**Time:** ~20-45 minutes

#### 3. Specific Test Categories

```bash
# Test only MCTS engine
python run_openai_tests.py --category TestMCTSEngineWithOpenAI --fast

# Test only fraud detection
python run_openai_tests.py --category TestFraudDetectionToolWithRealAPI --fast

# Test only classification
python run_openai_tests.py --category TestClassifyToolWithRealAPI --fast
```

#### 4. Dry Run (Validate Setup)

```bash
# Check configuration without running tests
python run_openai_tests.py --dry-run
```

### Direct pytest Usage

```bash
# Run all integration tests
pytest tests/test_openai_mcts_integration.py -v

# Run with specific model
pytest tests/test_openai_mcts_integration.py -v --model=o1-mini

# Run specific test class
pytest tests/test_openai_mcts_integration.py::TestMCTSEngineWithOpenAI -v

# Run specific test
pytest tests/test_openai_mcts_integration.py::TestMCTSEngineWithOpenAI::test_mcts_search_classification_real_api -v -s

# Skip slow tests
pytest tests/test_openai_mcts_integration.py -v -m "not slow"

# Run with verbose output
pytest tests/test_openai_mcts_integration.py -v -s --tb=short
```

## Test Configuration

### MCTS Configuration

Tests use two configurations:

**Fast (for development):**
```python
MCTSConfig(
    filter_iterations=10,
    classification_iterations=20,
    fraud_iterations=30,
    convergence_window=5
)
```

**Thorough (for validation):**
```python
MCTSConfig(
    filter_iterations=100,
    classification_iterations=500,
    fraud_iterations=1000,
    convergence_window=50
)
```

### Model Configuration

```python
LLMConfig(
    provider="openai",
    model="o1-mini",  # or "o1-preview", "o1", "o3-mini"
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1.0,  # Required for reasoning models
    max_tokens=4000
)
```

## What Gets Tested

### ✅ Covered

- [x] All 4 tools (filter, classify, fraud, CSV generation)
- [x] Multiple OpenAI reasoning models (o1-mini, o1-preview, o1, o3-mini)
- [x] Real API calls (NO MOCKS)
- [x] Synthetic transaction data with realistic patterns
- [x] Edge cases (tiny amounts, huge amounts, exact thresholds)
- [x] Currency conversion (GBP, USD, EUR, JPY)
- [x] Fraud detection across all risk levels (LOW, MEDIUM, HIGH, CRITICAL)
- [x] Classification across all categories
- [x] MCTS convergence and early termination
- [x] Performance benchmarks (latency, throughput)
- [x] Error handling (invalid data, API errors)
- [x] Batch processing
- [x] Fraud pattern detection (rapid succession, unusual amounts)
- [x] Model comparison

### ❌ Not Covered (by design)

- Mock LLM calls (explicitly excluded per requirements)
- Unit tests for algorithm logic (separate test file)
- Anthropic Claude tests (different test file)

## Understanding Test Output

### Successful Test Output

```
✓ Classification completed in 2.34s
  Hypothesis: {'category': 'Business', 'confidence': 0.85}
  Confidence: 0.85
  Nodes explored: 42

✓ Fraud detection completed in 3.12s
  Hypothesis: {'risk_level': 'HIGH', 'indicators': ['Large amount', 'Unusual merchant']}
  Confidence: 0.92

✓ Filtered 15 transactions above £250
  Total amount: £45,234.56
  Average amount: £3,015.64
```

### Test Metrics

Each test tracks:
- **Execution time** - Total time for MCTS search
- **Confidence scores** - Model confidence (0.0-1.0)
- **MCTS metadata**:
  - Root node visits (iterations)
  - Total nodes explored
  - Max depth reached
  - Final reward variance
  - Convergence status

### Performance Expectations

| Operation | Target Latency | Acceptable Range |
|-----------|---------------|------------------|
| Single classification | 2-5s | < 10s |
| Single fraud detection | 3-7s | < 15s |
| Batch (10 txns) | 30-60s | < 120s |
| Filter (100 txns) | < 1s | < 2s |

## Cost Management

### Estimated Costs (USD)

| Test Suite | o1-mini | o1-preview | o1 |
|------------|---------|------------|-----|
| Fast tests | $0.30 | $2.00 | $5.00 |
| Full suite | $1.50 | $10.00 | $25.00 |
| Single test | $0.02 | $0.15 | $0.40 |

### Cost Optimization Tips

1. **Use o1-mini for development** - 5-10x cheaper than o1-preview
2. **Run fast tests frequently** - Full suite only before commits
3. **Use pytest markers** - Skip slow/expensive tests: `-m "not slow"`
4. **Test specific categories** - Use `-k` flag to filter
5. **Reduce iterations** - Use fast MCTS config during development

### Monitoring Costs

```bash
# Run with cost tracking
pytest tests/test_openai_mcts_integration.py -v -s | grep "API calls"

# Count API calls per test
pytest tests/test_openai_mcts_integration.py -v --count-api-calls
```

## Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
❌ ERROR: OPENAI_API_KEY environment variable not set
```

**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
# Or add to .env file
```

#### 2. Rate Limits
```
openai.error.RateLimitError: Rate limit exceeded
```

**Solution:**
- Wait a few minutes
- Reduce test parallelization
- Use slower model (o1-mini)

#### 3. Timeout Errors
```
openai.error.Timeout: Request timed out
```

**Solution:**
- Increase timeout in config
- Reduce MCTS iterations
- Check network connection

#### 4. Model Not Available
```
ValueError: Model 'o3-mini' is not a reasoning model
```

**Solution:**
- Check model availability
- Update REASONING_MODELS list in config.py
- Use available model (o1-mini, o1-preview)

### Debug Mode

```bash
# Run with maximum verbosity
pytest tests/test_openai_mcts_integration.py -vvv -s --tb=long

# Enable logging
pytest tests/test_openai_mcts_integration.py -v --log-cli-level=DEBUG

# Run single test with debugging
pytest tests/test_openai_mcts_integration.py::TestMCTSEngineWithOpenAI::test_mcts_search_classification_real_api -vvv -s --pdb
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: OpenAI Integration Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run fast tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python run_openai_tests.py --fast --model o1-mini
```

## Test Data

### Sample Datasets

The synthetic data generator creates:

**Tiny (10 transactions)**: Quick smoke tests
**Small (100 transactions)**: Development and CI
**Medium (500 transactions)**: Pre-commit validation
**Large (1000 transactions)**: Release validation

### Data Characteristics

- **Categories**: Business (30%), Personal (30%), Travel (15%), Entertainment (10%), Investment (5%), Gambling (5%), Suspicious (5%)
- **Fraud Distribution**: LOW (80%), MEDIUM (12%), HIGH (6%), CRITICAL (2%)
- **Currencies**: GBP (70%), USD (15%), EUR (10%), JPY (5%)
- **Amount Ranges**: £0.01 to £500,000

## Contributing

### Adding New Tests

1. **Follow the pattern:**
```python
class TestNewFeature:
    """Test new feature with real OpenAI API."""

    def test_feature_basic(
        self, agent_config_openai, real_llm_client, synthetic_generator
    ):
        """Test basic functionality."""
        # Setup
        llm_function = create_llm_function(real_llm_client)

        # Execute
        result = your_feature(llm_function)

        # Verify
        assert result is not None
        print(f"\n✓ Feature test passed")
```

2. **Use fixtures** - Reuse `agent_config_openai`, `real_llm_client`, etc.
3. **Mark appropriately** - Add `@pytest.mark.slow` for expensive tests
4. **Print results** - Use `print(f"\n✓ ...")` for visibility
5. **Document cost** - Add cost estimate in docstring

### Test Checklist

Before submitting:
- [ ] Tests use real API (no mocks)
- [ ] Tests work with all models (o1-mini, o1-preview)
- [ ] Edge cases covered
- [ ] Performance benchmarks included
- [ ] Error handling tested
- [ ] Cost estimated and documented
- [ ] Output is clear and actionable

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Project README](README.md)
- [Requirements Specification](REQUIREMENTS.md)

## Support

For issues or questions:
1. Check this guide
2. Review test output carefully
3. Check OpenAI API status
4. Open an issue with:
   - Test command used
   - Full error output
   - API key status (redacted)
   - Model and configuration
