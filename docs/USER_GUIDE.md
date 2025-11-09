# User Guide - Financial Transaction Analysis Agent

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Understanding MCTS Reasoning](#understanding-mcts-reasoning)
6. [Interpreting Results](#interpreting-results)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- UV (recommended) or pip
- Valid API key for OpenAI or Anthropic

### Install with UV (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd financial-transaction-agent

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync
```

### Install with pip

```bash
# Clone the repository
git clone <repository-url>
cd financial-transaction-agent

# Install in editable mode
pip install -e .
```

---

## Quick Start

### 1. Set Up API Keys

Create a `.env` file in the project root:

```bash
# Copy example file
cp .env.example .env

# Edit with your API key
# For OpenAI
OPENAI_API_KEY=sk-your-key-here

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 2. Prepare Your CSV File

Your CSV file must have these columns:
- `transaction_id`: Unique identifier
- `amount`: Transaction amount (positive number)
- `currency`: Currency code (GBP, USD, EUR, etc.)
- `date`: Transaction date (YYYY-MM-DD format)
- `merchant`: Merchant name
- `description`: Transaction description
- `category`: (Optional) Transaction category

Example:
```csv
transaction_id,amount,currency,date,merchant,category,description
TX001,450.00,GBP,2025-01-15,Amazon UK,Business,Office supplies
TX002,1250.00,GBP,2025-01-17,British Airways,Travel,Flight to NYC
```

### 3. Run Analysis

```bash
uv run python -m src.cli analyze examples/sample_transactions.csv \
  --model o1-mini \
  --output results.csv
```

---

## Configuration

### LLM Provider Selection

#### OpenAI (Recommended for Cost)

```bash
uv run python -m src.cli analyze data.csv \
  --llm-provider openai \
  --model o1-mini \
  --api-key $OPENAI_API_KEY
```

**Supported Models:**
- `o1` (most powerful, expensive)
- `o1-preview` (preview version)
- `o1-mini` (recommended - fast and cost-effective)
- `o3-mini` (latest)

#### Anthropic Claude

```bash
uv run python -m src.cli analyze data.csv \
  --llm-provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --api-key $ANTHROPIC_API_KEY
```

**Supported Models:**
- `claude-3-5-sonnet-20241022`
- `claude-3-7-sonnet-20250219`
- `claude-sonnet-4-5-20250929`

### MCTS Configuration

Adjust MCTS reasoning depth:

```bash
uv run python -m src.cli analyze data.csv \
  --model o1-mini \
  --mcts-iterations 200  # Default: 100
```

**Guidelines:**
- **50-100 iterations**: Fast, good for testing
- **100-200 iterations**: Balanced (recommended)
- **200-500 iterations**: High accuracy, slower

### Threshold and Currency

Filter transactions above a specific amount:

```bash
uv run python -m src.cli analyze data.csv \
  --model o1-mini \
  --threshold 500.0 \
  --currency USD
```

---

## Usage Examples

### Example 1: Basic Analysis

```bash
uv run python -m src.cli analyze transactions.csv --model o1-mini
```

Output:
- Enhanced CSV: `enhanced_transactions.csv`
- Contains all original columns plus analysis results

### Example 2: Custom Threshold and Output Path

```bash
uv run python -m src.cli analyze transactions.csv \
  --model o1-mini \
  --threshold 1000 \
  --currency GBP \
  --output high_value_analysis.csv
```

### Example 3: High Accuracy Mode

```bash
uv run python -m src.cli analyze transactions.csv \
  --model o1 \
  --mcts-iterations 300 \
  --verbose
```

### Example 4: Validate CSV Before Processing

```bash
# Validate CSV format
uv run python -m src.cli validate transactions.csv

# If valid, run analysis
uv run python -m src.cli analyze transactions.csv --model o1-mini
```

### Example 5: List Available Models

```bash
uv run python -m src.cli models
```

---

## Understanding MCTS Reasoning

### What is MCTS?

Monte Carlo Tree Search (MCTS) is an algorithm that explores possible decisions by:
1. **Selection**: Choosing promising paths to explore
2. **Expansion**: Generating new hypotheses
3. **Simulation**: Testing hypotheses with the LLM
4. **Backpropagation**: Updating confidence scores

### How It Works for Transaction Analysis

#### Classification Example:
```
Transaction: £450 to Amazon UK for "Office supplies"

MCTS explores hypotheses:
1. "Business Expense - Office Supplies" (85% confidence)
2. "Personal Purchase - Mixed Use" (40% confidence)
3. "Recurring Business Expense" (70% confidence)

After 100 iterations:
→ Best: "Business Expense - Office Supplies" (confidence: 0.92)
```

#### Fraud Detection Example:
```
Transaction: £15,000 to "Wire Transfer - Tax Haven"

MCTS explores hypotheses:
1. Risk: CRITICAL, Indicators: [Large amount, Tax haven, Unusual] (95%)
2. Risk: HIGH, Indicators: [Large amount, Unknown merchant] (80%)
3. Risk: MEDIUM, Indicators: [Unusual timing] (30%)

After 100 iterations:
→ Best: CRITICAL risk (confidence: 0.96)
→ Actions: ["Immediate review", "Contact authorities", "Freeze account"]
```

### Benefits of MCTS
- **Explores multiple possibilities** before deciding
- **Balances exploration and exploitation**
- **Provides confidence scores** based on iterations
- **Explainable reasoning** through hypothesis tree

---

## Interpreting Results

### Enhanced CSV Columns

The output CSV includes these additional columns:

| Column | Description | Values |
|--------|-------------|--------|
| `above_250_gbp` | Transaction above threshold | `True` |
| `classification` | Transaction category | "Business Expense - Office Supplies" |
| `classification_confidence` | Classification certainty | 0.0 - 1.0 (0.92 = 92%) |
| `fraud_risk` | Fraud risk level | LOW / MEDIUM / HIGH / CRITICAL |
| `fraud_confidence` | Fraud detection certainty | 0.0 - 1.0 |
| `fraud_reasoning` | Detailed MCTS explanation | Text explanation |
| `mcts_iterations` | Total iterations performed | e.g., 200 |

### Risk Level Guidelines

- **LOW**: Normal transaction, no action needed
- **MEDIUM**: Review recommended, minor red flags
- **HIGH**: Immediate review required, multiple indicators
- **CRITICAL**: Urgent action needed, likely fraud

### Confidence Score Interpretation

- **0.9 - 1.0**: Very high confidence, reliable
- **0.7 - 0.9**: High confidence, likely correct
- **0.5 - 0.7**: Moderate confidence, review recommended
- **< 0.5**: Low confidence, manual review required

---

## Troubleshooting

### Issue: "Configuration error: Invalid API key"

**Solution:**
1. Check your `.env` file has the correct API key
2. Ensure no extra spaces or quotes in the key
3. Verify the key is valid on your provider's dashboard

```bash
# Check environment variable
echo $OPENAI_API_KEY

# Or use inline
uv run python -m src.cli analyze data.csv --model o1-mini --api-key sk-your-key
```

### Issue: "Model 'gpt-4' is not a reasoning model"

**Solution:**
Only reasoning models are supported. Use:
- OpenAI: `o1`, `o1-mini`, `o3-mini`
- Anthropic: `claude-3-5-sonnet-20241022`

```bash
# Correct
uv run python -m src.cli analyze data.csv --model o1-mini

# Incorrect
uv run python -m src.cli analyze data.csv --model gpt-4
```

### Issue: "CSV validation errors: Missing required columns"

**Solution:**
Ensure your CSV has all required columns:

```bash
# Validate first
uv run python -m src.cli validate data.csv

# Fix missing columns, then retry
uv run python -m src.cli analyze data.csv --model o1-mini
```

### Issue: "No transactions above threshold"

**Solution:**
Lower the threshold or check your currency conversion:

```bash
# Lower threshold
uv run python -m src.cli analyze data.csv --model o1-mini --threshold 50

# Or change currency
uv run python -m src.cli analyze data.csv --model o1-mini --currency USD
```

### Issue: Processing is very slow

**Solution:**
1. Reduce MCTS iterations for faster processing
2. Use a faster model (`o1-mini` instead of `o1`)
3. Process fewer transactions at a time

```bash
# Faster processing
uv run python -m src.cli analyze data.csv \
  --model o1-mini \
  --mcts-iterations 50
```

---

## FAQ

### Q: How much does it cost to run?

**A:** Cost depends on:
- LLM provider and model
- Number of transactions
- MCTS iterations

**Example with o1-mini:**
- 100 transactions × 100 iterations × 2 (classify + fraud) = ~20,000 LLM calls
- Estimated cost: $5-10 (varies by provider)

### Q: Can I use my own fraud detection rules?

**A:** Currently, fraud detection is purely LLM-based with MCTS reasoning. Custom rules can be added by modifying `src/mcts_engine.py`.

### Q: How accurate is the fraud detection?

**A:** Accuracy depends on:
- LLM model quality
- MCTS iterations
- Transaction context

For best results:
- Use `o1` model (most accurate)
- Set MCTS iterations to 200+
- Provide detailed transaction descriptions

### Q: Can I process real-time transactions?

**A:** The current version processes CSV files in batch mode. Real-time processing would require additional infrastructure (API server, streaming, etc.).

### Q: Is my transaction data secure?

**A:** Transaction data is sent to the LLM provider's API. For sensitive data:
- Use self-hosted LLMs
- Anonymize merchant names and descriptions
- Remove PII before processing

### Q: Can I use models other than OpenAI/Anthropic?

**A:** Currently, only OpenAI and Anthropic are supported. Additional providers can be added by modifying `src/config.py`.

### Q: What if I have millions of transactions?

**A:** For large datasets:
- Process in batches
- Use a more efficient model
- Consider parallel processing (custom implementation needed)
- Use a database instead of CSV

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
- Examples: [repository-url]/examples
