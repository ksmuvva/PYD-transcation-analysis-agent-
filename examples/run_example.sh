#!/bin/bash

# Example: Run financial transaction analysis
# This script demonstrates how to use the financial transaction analysis agent

echo "Financial Transaction Analysis Agent - Example Run"
echo "=================================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠ Warning: .env file not found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running."
    exit 1
fi

# Check if example CSV exists
if [ ! -f examples/sample_transactions.csv ]; then
    echo "⚠ Error: Sample transactions CSV not found"
    exit 1
fi

echo "Running analysis on sample transactions..."
echo ""

# Run analysis with o1-mini (cost-effective reasoning model)
python -m src.cli analyze examples/sample_transactions.csv \
    --model o1-mini \
    --llm-provider openai \
    --threshold 250 \
    --currency GBP \
    --mcts-iterations 50 \
    --output examples/output_enhanced.csv \
    --verbose

echo ""
echo "✓ Analysis complete!"
echo "Check examples/output_enhanced.csv for results"
