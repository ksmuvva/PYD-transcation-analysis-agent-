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

## Documentation

- [Requirements Specification](REQUIREMENTS.md) - Detailed requirements
- [System Design](DESIGN.md) - Architecture and component design
- [Implementation TODO](TODO.md) - Development roadmap

## Technology Stack

- **Pydantic AI**: Agent orchestration and tool management
- **MCTS Engine**: Custom Monte Carlo Tree Search implementation
- **Pandas**: CSV data processing
- **Typer + Rich**: Beautiful CLI interface
- **OpenAI/Anthropic**: LLM reasoning capabilities

## Project Status

🚧 **In Development** - See [TODO.md](TODO.md) for implementation progress

## License

MIT