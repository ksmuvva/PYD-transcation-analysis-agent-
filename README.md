# Financial Transaction Analysis Agent with MCTS Reasoning

An AI-powered financial transaction analysis agent built with **Pydantic AI** that leverages **Monte Carlo Tree Search (MCTS)** reasoning to classify transactions and detect fraud.

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

```bash
# Install dependencies
poetry install

# Set up your API key
export OPENAI_API_KEY="your-key-here"

# Run analysis
python -m src.cli analyze \
  --csv-file transactions.csv \
  --llm-provider openai \
  --model o1-mini \
  --threshold 250 \
  --currency GBP
```

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
poetry run financial-agent analyze examples/sample_transactions.csv --model o1-mini

# Traces are automatically logged to console and sent to Logfire
```

To disable telemetry:
```bash
poetry run financial-agent analyze examples/sample_transactions.csv --model o1-mini --no-telemetry
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

## Documentation

- [User Guide](docs/USER_GUIDE.md) - How to use the agent
- [Requirements Specification](REQUIREMENTS.md) - Detailed requirements
- [System Design](DESIGN.md) - Architecture and component design
- [Implementation TODO](TODO.md) - Development roadmap

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