# Financial Transaction Analysis Agent with MCTS Reasoning

An AI-powered financial transaction analysis agent built with **Pydantic AI** that leverages **Monte Carlo Tree Search (MCTS)** reasoning to classify transactions and detect fraud.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (required for Pydantic AI)
- **uv** - Fast Python package installer ([installation guide](https://github.com/astral-sh/uv))
- **API Key** from either:
  - OpenAI ([get API key](https://platform.openai.com/api-keys))
  - Anthropic Claude ([get API key](https://console.anthropic.com/))

## Features

- **MCTS-Powered Reasoning**: Advanced Monte Carlo Tree Search algorithm for intelligent transaction classification and fraud detection
- **Multi-LLM Support**: Compatible with OpenAI (o1, o1-mini, o3-mini) and Anthropic Claude reasoning models
- **4 Specialized Tools**:
  1. Filter transactions above 250 GBP (with currency conversion)
  2. Classify transactions using MCTS reasoning
  3. Detect fraudulent transactions using MCTS reasoning
  4. Generate enhanced CSV with analysis results
- **Full Observability**: Integrated Pydantic Logfire telemetry for comprehensive tracing and evaluation
- **CLI Interface**: User-friendly command-line interface with progress tracking
- **Type-Safe**: Built with Pydantic AI for complete type safety
- **Explainable AI**: Detailed reasoning traces for all decisions

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd PYD-transcation-analysis-agent-

# Install dependencies with uv
uv sync
```

### 2. Set Up API Keys

**IMPORTANT:** API keys must be set via environment variables for security. Do not pass them as command-line arguments.

For OpenAI:
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
```

For Pydantic Logfire (optional, for telemetry):
```bash
export LOGFIRE_TOKEN="your-logfire-token-here"
```

### 3. Prepare Your CSV File

Your CSV file must contain the following columns:
- `transaction_id` - Unique identifier for each transaction
- `amount` - Transaction amount (positive number)
- `currency` - Currency code (GBP, USD, EUR, JPY, CAD, AUD)
- `date` - Transaction date (YYYY-MM-DD format recommended)
- `merchant` - Merchant name
- `description` - Transaction description

Example CSV format:
```csv
transaction_id,amount,currency,date,merchant,description
TX001,350.00,GBP,2025-01-15,Amazon UK,Office supplies
TX002,500.00,USD,2025-01-16,Dell,Laptop purchase
```

See `examples/sample_transactions.csv` for a complete example.

### 4. Run Analysis

```bash
# Basic analysis with OpenAI
uv run python -m src.cli analyze \
  examples/sample_transactions.csv \
  --llm-provider openai \
  --model o1-mini

# With custom threshold and currency
uv run python -m src.cli analyze \
  examples/sample_transactions.csv \
  --llm-provider openai \
  --model o1-mini \
  --threshold 250 \
  --currency GBP \
  --output results.csv

# Using Anthropic Claude
uv run python -m src.cli analyze \
  examples/sample_transactions.csv \
  --llm-provider anthropic \
  --model claude-3-5-sonnet-20241022

# With verbose output and custom MCTS iterations
uv run python -m src.cli analyze \
  examples/sample_transactions.csv \
  --model o1-mini \
  --mcts-iterations 200 \
  --verbose

# Without telemetry (for faster processing)
uv run python -m src.cli analyze \
  examples/sample_transactions.csv \
  --model o1-mini \
  --no-telemetry
```

### 5. View Results

The enhanced CSV file will be created with additional columns containing:
- Classification results
- Fraud risk assessments
- Confidence scores
- MCTS reasoning explanations

## Output

The agent produces an enhanced CSV with additional columns:
- `above_250_gbp`: Boolean flag for transactions >= 250 GBP
- `classification`: Transaction category (Business, Personal, etc.)
- `classification_confidence`: Confidence score (0-1)
- `fraud_risk`: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
- `fraud_confidence`: Fraud detection confidence (0-1)
- `fraud_reasoning`: MCTS reasoning explanation
- `mcts_iterations`: Number of MCTS iterations performed

## Pydantic Logfire Observability

This agent includes comprehensive telemetry powered by **Pydantic Logfire**:

### What Gets Tracked

- ✅ **All LLM Calls**: Prompts, responses, tokens, latency, costs
- ✅ **Agent Operations**: Tool calls, reasoning steps, dependencies
- ✅ **MCTS Details**: Iterations, node exploration, convergence patterns
- ✅ **Transaction Analysis**: Classification results, fraud detection, confidence scores
- ✅ **Performance Metrics**: Processing time, token usage, error rates

### Quick Start with Logfire

Logfire telemetry is enabled by default. The agent automatically traces all operations:

```bash
uv run financial-agent analyze examples/sample_transactions.csv --model o1-mini

# Traces are automatically logged to console and sent to Logfire
```

To disable telemetry:
```bash
uv run financial-agent analyze examples/sample_transactions.csv --model o1-mini --no-telemetry
```

### View Your Traces

Visit [https://logfire.pydantic.dev](https://logfire.pydantic.dev) to:
- Explore detailed traces of every agent run
- Analyze MCTS search behavior and convergence
- Track token usage, costs, and performance metrics
- Debug classification and fraud detection decisions
- Monitor real-time agent performance
- Export trace data for offline analysis

### Configuration

Set `LOGFIRE_TOKEN` in your `.env` file to send traces to the Logfire platform. Without a token, traces will only be logged to the console.

## CLI Commands

### Analyze Transactions
```bash
uv run python -m src.cli analyze <csv_file> [OPTIONS]
```

Options:
- `--output, -o` - Output file path (default: enhanced_transactions.csv)
- `--llm-provider, -p` - LLM provider: openai or anthropic (default: openai)
- `--model, -m` - Model name (required, e.g., o1-mini, claude-3-5-sonnet-20241022)
- `--threshold, -t` - Transaction threshold amount (default: 250.0)
- `--currency, -c` - Base currency for filtering (default: GBP)
- `--mcts-iterations, -i` - Number of MCTS iterations per transaction (default: 100)
- `--verbose, -v` - Enable verbose output
- `--telemetry/--no-telemetry` - Enable/disable Pydantic Logfire telemetry (default: enabled)

### Validate CSV File
```bash
uv run python -m src.cli validate <csv_file>
```

Validates your CSV file format without running analysis.

### List Available Models
```bash
uv run python -m src.cli models
```

Shows all available reasoning models for each provider.

## Troubleshooting

### "Configuration error: Missing API key"
Make sure you've exported the API key as an environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### "CSV validation errors"
Run the validate command to see specific errors:
```bash
uv run python -m src.cli validate your_file.csv
```

### "No transactions above threshold"
Lower the threshold value or check your currency conversion:
```bash
uv run python -m src.cli analyze your_file.csv --threshold 100 --currency USD
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) - How to use the agent
- [Requirements Specification](REQUIREMENTS.md) - Detailed requirements
- [System Design](DESIGN.md) - Architecture and component design (complex system)
- [Implementation TODO](TODO.md) - Development roadmap
- [Testing Guide](TESTING_GUIDE.md) - How to run tests
- [Defects Log](defects.md) - Known issues and fixes

## Technology Stack

- **Pydantic AI**: Agent orchestration and tool management
- **Pydantic Logfire**: Observability, tracing, and telemetry
- **MCTS Engine**: Custom Monte Carlo Tree Search implementation
- **Pandas**: CSV data processing
- **Typer + Rich**: Beautiful CLI interface
- **OpenAI/Anthropic**: LLM reasoning capabilities

## Project Status

🚧 **In Development** - See [TODO.md](TODO.md) for implementation progress

## License

MIT