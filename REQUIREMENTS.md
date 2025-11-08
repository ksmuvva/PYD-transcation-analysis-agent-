# Financial Transaction Analysis Agent - Requirements Specification

## 1. Overview
Build an AI-powered financial transaction analysis agent using **Pydantic AI** that processes CSV files containing financial transactions, applies MCTS (Monte Carlo Tree Search) reasoning for analysis, and outputs enhanced CSV with fraud detection and transaction classification.

## 2. Core Requirements

### 2.1 Framework & Technology Stack
- **Primary Framework**: Pydantic AI for agent orchestration
- **Language**: Python 3.10+
- **Data Processing**: pandas for CSV operations
- **AI/LLM Integration**: Multi-provider support (OpenAI, Anthropic, etc.)
- **Reasoning Engine**: MCTS (Monte Carlo Tree Search) reasoning
- **CLI Framework**: Click or Typer for command-line interface

### 2.2 LLM Provider Requirements
- **Supported Providers**:
  - OpenAI (o1, o1-mini, o3-mini - reasoning models)
  - Anthropic Claude (Claude 3.5 Sonnet with extended thinking)
  - Other reasoning-capable models
- **Configuration**:
  - User provides API keys via CLI or environment variables
  - User selects LLM provider and model at runtime
  - Validation: Only reasoning models allowed
- **API Key Management**: Secure handling, no hardcoding

### 2.3 Input Requirements
- **Format**: CSV file
- **Required Columns**:
  - `transaction_id`: Unique identifier
  - `amount`: Transaction amount (numeric)
  - `currency`: Currency code (e.g., GBP, USD)
  - `date`: Transaction date
  - `merchant`: Merchant name
  - `category`: Transaction category (optional)
  - `description`: Transaction description
- **Upload Method**: CLI file path parameter
- **Validation**:
  - File exists and is readable
  - Valid CSV format
  - Required columns present
  - Data type validation

### 2.4 Output Requirements
- **Format**: CSV file
- **Original Columns**: All input columns preserved
- **Additional Columns**:
  - `above_250_gbp`: Boolean (True if amount >= 250 GBP equivalent)
  - `classification`: Transaction category/type (from Tool 2)
  - `classification_confidence`: Confidence score (0-1)
  - `fraud_risk`: Fraud risk level (LOW/MEDIUM/HIGH/CRITICAL)
  - `fraud_confidence`: Fraud detection confidence (0-1)
  - `fraud_reasoning`: MCTS reasoning explanation
  - `mcts_iterations`: Number of MCTS iterations performed
- **Filtering**: Only transactions >= 250 GBP (or equivalent) in final output

## 3. Agent Tools Specification

### 3.1 Tool 1: Transaction Filter
**Name**: `filter_transactions_above_threshold`

**Purpose**: Extract all transactions above 250 GBP (or equivalent in other currencies)

**Inputs**:
- CSV DataFrame
- Threshold amount (default: 250)
- Base currency (default: GBP)

**Outputs**:
- Filtered DataFrame with transactions >= threshold
- Count of filtered transactions
- Summary statistics

**Requirements**:
- Currency conversion support
- Configurable threshold
- Statistical summary

### 3.2 Tool 2: Transaction Classification with MCTS
**Name**: `classify_transactions_mcts`

**Purpose**: Classify transactions into categories using MCTS reasoning

**MCTS Reasoning Strategy**:
- **Selection**: Choose promising transaction patterns
- **Expansion**: Generate possible classification hypotheses
- **Simulation**: Test classification against transaction features
- **Backpropagation**: Update classification confidence

**Classification Categories**:
- Personal/Business
- Essential/Discretionary
- Recurring/One-time
- Risk level (Low/Medium/High)
- Custom categories based on merchant/description

**Inputs**:
- Transaction record
- Historical transaction context
- MCTS configuration (iterations, depth)

**Outputs**:
- Primary classification
- Confidence score
- Alternative classifications (top 3)
- MCTS reasoning trace

### 3.3 Tool 3: Fraud Detection with MCTS
**Name**: `detect_fraud_mcts`

**Purpose**: Identify potentially fraudulent transactions using MCTS reasoning

**MCTS Reasoning Strategy**:
- **Selection**: Focus on suspicious transaction patterns
- **Expansion**: Generate fraud hypotheses (unusual amount, location, timing, etc.)
- **Simulation**: Test fraud indicators against transaction history
- **Backpropagation**: Update fraud risk scores

**Fraud Indicators**:
- Unusual transaction amounts
- Abnormal merchant patterns
- Suspicious timing/frequency
- Geographic anomalies (if location data available)
- Velocity checks (rapid succession)
- Pattern deviations from user history

**Inputs**:
- Transaction record
- User transaction history
- MCTS configuration
- Fraud detection rules/patterns

**Outputs**:
- Fraud risk level (LOW/MEDIUM/HIGH/CRITICAL)
- Confidence score (0-1)
- Detected fraud indicators
- MCTS reasoning explanation
- Recommended actions

### 3.4 Tool 4: CSV Report Generator
**Name**: `generate_enhanced_csv`

**Purpose**: Create final CSV with all analysis results

**Inputs**:
- Original CSV data
- Tool 1 results (filtered transactions)
- Tool 2 results (classifications)
- Tool 3 results (fraud detections)

**Outputs**:
- Enhanced CSV file with all additional columns
- Summary report (JSON/text)
- Processing statistics

**Requirements**:
- Preserve data integrity
- Handle missing/null values
- Generate human-readable report
- Output file naming convention

## 4. Pydantic AI Agent Architecture

### 4.1 Agent Design
```python
# Pseudo-structure
class FinancialTransactionAgent:
    - LLM provider configuration
    - MCTS reasoning engine
    - Transaction state management
    - Tool orchestration
    - Result aggregation
```

### 4.2 Agent State (Dependencies)
- Uploaded CSV data (shared across all tools)
- User configuration (threshold, currency, etc.)
- LLM client instance
- MCTS parameters
- Processing metadata

### 4.3 Agent Tools Integration
- All 4 tools registered as Pydantic AI tools
- Tools can access shared state via dependencies
- Sequential execution: Tool 1 → Tool 2 → Tool 3 → Tool 4
- Each tool returns structured Pydantic models

### 4.4 Result Models
```python
# Pydantic models for structured outputs
- TransactionFilterResult
- ClassificationResult
- FraudDetectionResult
- EnhancedCSVResult
```

## 5. CLI Interface Requirements

### 5.1 Command Structure
```bash
python agent_cli.py analyze \
  --csv-file <path_to_csv> \
  --output <output_csv_path> \
  --llm-provider <openai|anthropic|other> \
  --model <model_name> \
  --api-key <api_key_or_env_var> \
  --threshold <amount> \
  --currency <GBP|USD|EUR> \
  --mcts-iterations <number>
```

### 5.2 CLI Features
- **Interactive Mode**: Prompt for missing parameters
- **Configuration File Support**: YAML/JSON config file
- **Validation**: Pre-flight checks before processing
- **Progress Indicators**: Show processing status
- **Verbose Mode**: Detailed logging
- **Dry Run**: Validate without processing

### 5.3 LLM Provider Selection
- Interactive menu to choose provider
- Model validation (reasoning models only)
- API key validation before processing
- Support for environment variables (e.g., `OPENAI_API_KEY`)

## 6. MCTS Reasoning Engine

### 6.1 Core Components
- **State Representation**: Transaction features + context
- **Action Space**: Classification/fraud hypotheses
- **Simulation Policy**: LLM-guided reasoning
- **Evaluation Function**: Confidence scoring
- **Selection Strategy**: UCB1 (Upper Confidence Bound)

### 6.2 Configuration
- Configurable iterations (default: 100)
- Exploration constant (C) tuning
- Maximum tree depth
- Simulation budget per node
- Parallel simulations support

### 6.3 LLM Integration
- LLM acts as simulation policy
- Reasoning model generates hypotheses
- Structured output parsing
- Confidence calibration

## 7. Non-Functional Requirements

### 7.1 Performance
- Process 1000 transactions in < 5 minutes (depends on LLM latency)
- Batch processing for LLM calls
- Efficient MCTS tree pruning
- Memory-efficient CSV handling

### 7.2 Reliability
- Error handling and recovery
- Input validation
- Graceful degradation
- Transaction atomicity

### 7.3 Security
- Secure API key handling
- No sensitive data logging
- PII protection (if applicable)
- Input sanitization

### 7.4 Maintainability
- Clean code structure
- Comprehensive documentation
- Type hints throughout
- Unit tests for tools
- Integration tests

### 7.5 Usability
- Clear error messages
- Helpful CLI help text
- Example CSV files
- Quick start guide

## 8. Constraints & Assumptions

### 8.1 Constraints
- **Strictly Pydantic AI**: Must use Pydantic AI framework
- **Reasoning Models Only**: No standard completion models
- **CSV Format**: Only CSV input/output (no Excel, JSON, etc.)
- **Single Currency Processing**: One base currency per run

### 8.2 Assumptions
- Users have valid LLM API keys
- CSV files are reasonably sized (< 100K rows)
- Transaction data is in English
- Users understand MCTS concepts (basic level)
- Internet connection available for LLM API calls

## 9. Success Criteria

### 9.1 Functional Success
- ✅ All 4 tools working correctly
- ✅ MCTS reasoning integrated in Tools 2 & 3
- ✅ Accurate transaction filtering (>= 250 GBP)
- ✅ Meaningful classification results
- ✅ Fraud detection with explanations
- ✅ Valid enhanced CSV output

### 9.2 Technical Success
- ✅ Follows Pydantic AI best practices
- ✅ Type-safe throughout
- ✅ Comprehensive error handling
- ✅ Clean separation of concerns
- ✅ Well-documented code

### 9.3 User Success
- ✅ Easy CLI usage
- ✅ Clear output and reports
- ✅ Helpful error messages
- ✅ Reasonable processing time
- ✅ Actionable fraud insights

## 10. Future Enhancements (Out of Scope for V1)
- Web UI interface
- Real-time transaction processing
- Multi-currency batch processing
- Custom fraud rule configuration
- Machine learning model integration
- Historical trend analysis
- API server mode
- Database integration
- Email/Slack notifications
