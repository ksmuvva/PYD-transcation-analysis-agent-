# Financial Transaction Analysis Agent - System Design

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│  (Typer-based: User inputs, LLM selection, CSV upload)          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Configuration Manager                          │
│  - API Key validation                                            │
│  - LLM Provider setup (OpenAI/Anthropic/etc.)                   │
│  - MCTS parameters                                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Pydantic AI Agent (Main Orchestrator)              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Agent Dependencies (State)                   │  │
│  │  - CSV DataFrame                                          │  │
│  │  - LLM Client                                             │  │
│  │  - MCTS Engine                                            │  │
│  │  - Configuration                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Agent Tools                            │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ Tool 1: filter_transactions_above_threshold      │    │  │
│  │  │ - Currency conversion                             │    │  │
│  │  │ - Threshold filtering                             │    │  │
│  │  │ - Statistics generation                           │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ Tool 2: classify_transactions_mcts                │    │  │
│  │  │ - MCTS reasoning for classification               │    │  │
│  │  │ - Pattern detection                               │    │  │
│  │  │ - Confidence scoring                              │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ Tool 3: detect_fraud_mcts                         │    │  │
│  │  │ - MCTS reasoning for fraud detection              │    │  │
│  │  │ - Anomaly detection                               │    │  │
│  │  │ - Risk scoring                                    │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────┐    │  │
│  │  │ Tool 4: generate_enhanced_csv                     │    │  │
│  │  │ - Results aggregation                             │    │  │
│  │  │ - CSV generation                                  │    │  │
│  │  │ - Report creation                                 │    │  │
│  │  └──────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCTS Reasoning Engine                         │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Selection  │→ │  Expansion  │→ │ Simulation  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         ↑                                  │                    │
│         │                                  ▼                    │
│  ┌──────────────────────────────────────────────────┐          │
│  │            Backpropagation                        │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                  │
│  - LLM-guided reasoning                                         │
│  - Tree-based search                                            │
│  - Confidence scoring                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Provider Layer                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   OpenAI     │  │  Anthropic   │  │    Other     │         │
│  │  (o1, o1-   │  │  (Claude 3.5 │  │   Reasoning  │         │
│  │   mini, o3)  │  │   Sonnet)    │  │    Models    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Layer                                │
│  - Enhanced CSV file                                             │
│  - JSON summary report                                           │
│  - Processing logs                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Design

### 2.1 CLI Module (`cli.py`)
**Purpose**: User interface and input handling

**Key Functions**:
```python
def main():
    """Main CLI entry point"""

def analyze_command(
    csv_file: Path,
    output: Path,
    llm_provider: str,
    model: str,
    api_key: str,
    threshold: float,
    currency: str,
    mcts_iterations: int
):
    """Main analysis command"""

def validate_inputs(csv_file: Path, llm_provider: str, model: str):
    """Validate all inputs before processing"""

def interactive_llm_selection():
    """Interactive menu for LLM provider selection"""
```

**Dependencies**: Typer, Rich (for beautiful CLI output)

---

### 2.2 Configuration Module (`config.py`)
**Purpose**: Manage configuration and LLM provider setup

**Key Classes**:
```python
@dataclass
class LLMConfig:
    provider: str  # "openai", "anthropic", etc.
    model: str
    api_key: str
    temperature: float = 0.0  # Reasoning models work best at low temp
    max_tokens: int = 4000

@dataclass
class MCTSConfig:
    iterations: int = 100
    exploration_constant: float = 1.414  # sqrt(2) for UCB1
    max_depth: int = 5
    simulation_budget: int = 10

@dataclass
class AgentConfig:
    llm: LLMConfig
    mcts: MCTSConfig
    threshold_amount: float = 250.0
    base_currency: str = "GBP"

class ConfigManager:
    """Validates and manages all configuration"""

    @staticmethod
    def validate_reasoning_model(provider: str, model: str) -> bool:
        """Ensure only reasoning models are used"""

    @staticmethod
    def create_llm_client(config: LLMConfig):
        """Create Pydantic AI compatible LLM client"""
```

**Reasoning Models Whitelist**:
- OpenAI: `o1`, `o1-mini`, `o1-preview`, `o3-mini`
- Anthropic: `claude-3-5-sonnet-20241022` (with extended thinking)
- Others: TBD based on availability

---

### 2.3 Data Models (`models.py`)
**Purpose**: Pydantic models for type safety and validation

**Key Models**:
```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class Currency(str, Enum):
    GBP = "GBP"
    USD = "USD"
    EUR = "EUR"
    # Add more as needed

class Transaction(BaseModel):
    """Input transaction model"""
    transaction_id: str
    amount: float = Field(gt=0)
    currency: Currency
    date: datetime
    merchant: str
    category: str | None = None
    description: str

class TransactionFilterResult(BaseModel):
    """Result from Tool 1"""
    filtered_count: int
    total_amount: float
    currency: Currency
    transactions: list[Transaction]

class ClassificationResult(BaseModel):
    """Result from Tool 2 (per transaction)"""
    transaction_id: str
    primary_classification: str
    confidence: float = Field(ge=0, le=1)
    alternative_classifications: list[tuple[str, float]]
    mcts_iterations: int
    reasoning_trace: str

class FraudRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class FraudDetectionResult(BaseModel):
    """Result from Tool 3 (per transaction)"""
    transaction_id: str
    risk_level: FraudRiskLevel
    confidence: float = Field(ge=0, le=1)
    detected_indicators: list[str]
    reasoning: str
    mcts_iterations: int
    recommended_actions: list[str]

class EnhancedTransaction(Transaction):
    """Output transaction with analysis results"""
    above_250_gbp: bool
    classification: str
    classification_confidence: float
    fraud_risk: FraudRiskLevel
    fraud_confidence: float
    fraud_reasoning: str
    mcts_iterations: int

class ProcessingReport(BaseModel):
    """Summary report"""
    total_transactions_analyzed: int
    transactions_above_threshold: int
    high_risk_transactions: int
    processing_time_seconds: float
    llm_provider: str
    model_used: str
    mcts_iterations_total: int
```

---

### 2.4 MCTS Engine (`mcts_engine.py`)
**Purpose**: Core MCTS reasoning implementation

**Key Classes**:
```python
from dataclasses import dataclass
from typing import Any, Callable
import math

@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree"""
    state: Any
    parent: 'MCTSNode | None' = None
    children: list['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0

    def ucb1_score(self, exploration_constant: float) -> float:
        """Upper Confidence Bound formula"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

class MCTSEngine:
    """MCTS reasoning engine with LLM integration"""

    def __init__(self, config: MCTSConfig, llm_function: Callable):
        self.config = config
        self.llm_function = llm_function  # Pydantic AI agent run
        self.root = None

    def search(self, initial_state: dict, objective: str) -> dict:
        """
        Main MCTS search loop

        Args:
            initial_state: Starting state (transaction data + context)
            objective: "classify" or "detect_fraud"

        Returns:
            Best result with confidence and reasoning
        """
        self.root = MCTSNode(state=initial_state)

        for _ in range(self.config.iterations):
            # 1. Selection
            node = self._select(self.root)

            # 2. Expansion
            if not node.children and node.visits > 0:
                node = self._expand(node, objective)

            # 3. Simulation (LLM-guided)
            reward = self._simulate(node, objective)

            # 4. Backpropagation
            self._backpropagate(node, reward)

        # Return best child
        best_child = max(self.root.children, key=lambda n: n.visits)
        return self._extract_result(best_child)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using UCB1"""
        while node.children:
            node = max(node.children,
                      key=lambda n: n.ucb1_score(self.config.exploration_constant))
        return node

    def _expand(self, node: MCTSNode, objective: str) -> MCTSNode:
        """Expand node with LLM-generated hypotheses"""
        # Use LLM to generate possible hypotheses
        hypotheses = self._generate_hypotheses(node.state, objective)

        # Create child nodes for each hypothesis
        node.children = [
            MCTSNode(state={**node.state, 'hypothesis': h}, parent=node)
            for h in hypotheses
        ]

        return node.children[0] if node.children else node

    def _simulate(self, node: MCTSNode, objective: str) -> float:
        """Simulate using LLM reasoning"""
        # Use LLM to evaluate the hypothesis
        result = self._evaluate_hypothesis(node.state, objective)
        return result['confidence']  # 0-1 score

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _generate_hypotheses(self, state: dict, objective: str) -> list[dict]:
        """Use LLM to generate hypotheses"""
        # Call LLM with structured prompt
        pass

    def _evaluate_hypothesis(self, state: dict, objective: str) -> dict:
        """Use LLM to evaluate a hypothesis"""
        # Call LLM with evaluation prompt
        pass

    def _extract_result(self, node: MCTSNode) -> dict:
        """Extract final result from best node"""
        return {
            'hypothesis': node.state.get('hypothesis'),
            'confidence': node.value / node.visits if node.visits > 0 else 0,
            'visits': node.visits,
            'reasoning': node.state.get('reasoning', '')
        }
```

---

### 2.5 Pydantic AI Agent (`agent.py`)
**Purpose**: Main agent orchestration with tools

**Key Components**:
```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
import pandas as pd

# Define dependencies (shared state)
@dataclass
class AgentDependencies:
    """Shared state across all tools"""
    df: pd.DataFrame
    config: AgentConfig
    mcts_engine: MCTSEngine
    llm_client: Model
    results: dict = None  # Store intermediate results

# Create the agent
financial_agent = Agent(
    model=None,  # Set dynamically based on config
    deps_type=AgentDependencies,
    system_prompt="""You are a financial transaction analysis expert.
    You use Monte Carlo Tree Search (MCTS) reasoning to analyze transactions,
    classify them accurately, and detect potential fraud.

    Always provide detailed reasoning for your conclusions.
    Consider multiple hypotheses before making final decisions.
    Use confidence scores to reflect uncertainty."""
)

# Tool 1: Filter transactions
@financial_agent.tool
async def filter_transactions_above_threshold(
    ctx: RunContext[AgentDependencies],
    threshold: float | None = None,
    currency: str | None = None
) -> TransactionFilterResult:
    """
    Filter transactions above specified threshold.

    Args:
        threshold: Amount threshold (default from config)
        currency: Base currency (default from config)
    """
    threshold = threshold or ctx.deps.config.threshold_amount
    currency = currency or ctx.deps.config.base_currency

    df = ctx.deps.df

    # Convert all amounts to base currency
    # (Currency conversion logic here)

    # Filter
    filtered_df = df[df['amount_gbp'] >= threshold]

    result = TransactionFilterResult(
        filtered_count=len(filtered_df),
        total_amount=filtered_df['amount_gbp'].sum(),
        currency=currency,
        transactions=[Transaction(**row) for row in filtered_df.to_dict('records')]
    )

    # Store in dependencies for next tools
    ctx.deps.results = {'filtered_transactions': filtered_df}

    return result

# Tool 2: Classify with MCTS
@financial_agent.tool
async def classify_transactions_mcts(
    ctx: RunContext[AgentDependencies],
    batch_size: int = 10
) -> list[ClassificationResult]:
    """
    Classify transactions using MCTS reasoning.

    Processes filtered transactions in batches for efficiency.
    """
    filtered_df = ctx.deps.results.get('filtered_transactions')
    if filtered_df is None:
        raise ValueError("Must run filter_transactions_above_threshold first")

    results = []
    mcts = ctx.deps.mcts_engine

    for idx, row in filtered_df.iterrows():
        transaction = Transaction(**row)

        # Prepare state for MCTS
        state = {
            'transaction': transaction.model_dump(),
            'context': {
                'historical_transactions': [],  # Could add user history
                'merchant_patterns': {},
            }
        }

        # Run MCTS search for classification
        mcts_result = mcts.search(state, objective='classify')

        result = ClassificationResult(
            transaction_id=transaction.transaction_id,
            primary_classification=mcts_result['hypothesis']['category'],
            confidence=mcts_result['confidence'],
            alternative_classifications=mcts_result.get('alternatives', []),
            mcts_iterations=ctx.deps.config.mcts.iterations,
            reasoning_trace=mcts_result['reasoning']
        )

        results.append(result)

    # Store for next tool
    ctx.deps.results['classifications'] = results

    return results

# Tool 3: Fraud detection with MCTS
@financial_agent.tool
async def detect_fraud_mcts(
    ctx: RunContext[AgentDependencies]
) -> list[FraudDetectionResult]:
    """
    Detect fraudulent transactions using MCTS reasoning.
    """
    filtered_df = ctx.deps.results.get('filtered_transactions')
    if filtered_df is None:
        raise ValueError("Must run filter_transactions_above_threshold first")

    results = []
    mcts = ctx.deps.mcts_engine

    for idx, row in filtered_df.iterrows():
        transaction = Transaction(**row)

        # Prepare state for MCTS
        state = {
            'transaction': transaction.model_dump(),
            'context': {
                'transaction_history': [],  # User's history
                'fraud_indicators': {},
            }
        }

        # Run MCTS search for fraud detection
        mcts_result = mcts.search(state, objective='detect_fraud')

        result = FraudDetectionResult(
            transaction_id=transaction.transaction_id,
            risk_level=mcts_result['hypothesis']['risk_level'],
            confidence=mcts_result['confidence'],
            detected_indicators=mcts_result['hypothesis'].get('indicators', []),
            reasoning=mcts_result['reasoning'],
            mcts_iterations=ctx.deps.config.mcts.iterations,
            recommended_actions=mcts_result.get('actions', [])
        )

        results.append(result)

    # Store for final tool
    ctx.deps.results['fraud_detections'] = results

    return results

# Tool 4: Generate enhanced CSV
@financial_agent.tool
async def generate_enhanced_csv(
    ctx: RunContext[AgentDependencies],
    output_path: str
) -> ProcessingReport:
    """
    Generate final CSV with all analysis results.
    """
    filtered_df = ctx.deps.results.get('filtered_transactions')
    classifications = ctx.deps.results.get('classifications', [])
    fraud_detections = ctx.deps.results.get('fraud_detections', [])

    # Merge all results
    # Create enhanced transactions
    # Generate CSV
    # Create summary report

    # (Implementation details)

    return ProcessingReport(
        total_transactions_analyzed=len(filtered_df),
        transactions_above_threshold=len(filtered_df),
        high_risk_transactions=sum(1 for f in fraud_detections if f.risk_level in ['HIGH', 'CRITICAL']),
        processing_time_seconds=0,  # Track actual time
        llm_provider=ctx.deps.config.llm.provider,
        model_used=ctx.deps.config.llm.model,
        mcts_iterations_total=len(filtered_df) * ctx.deps.config.mcts.iterations * 2  # classify + fraud
    )
```

---

### 2.6 CSV Processor (`csv_processor.py`)
**Purpose**: Handle CSV I/O and data validation

**Key Functions**:
```python
class CSVProcessor:
    """Handle CSV operations"""

    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """Load and validate CSV"""

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> bool:
        """Validate required columns exist"""

    @staticmethod
    def save_enhanced_csv(
        df: pd.DataFrame,
        classifications: list[ClassificationResult],
        fraud_detections: list[FraudDetectionResult],
        output_path: Path
    ):
        """Save enhanced CSV with all results"""
```

---

## 3. Data Flow

### 3.1 Processing Pipeline
```
1. CSV Upload
   ↓
2. Load & Validate CSV → DataFrame
   ↓
3. Initialize Agent Dependencies (LLM, MCTS, Config)
   ↓
4. Tool 1: Filter transactions >= 250 GBP
   ↓
5. Tool 2: Classify each transaction (MCTS)
   ↓
6. Tool 3: Detect fraud for each transaction (MCTS)
   ↓
7. Tool 4: Merge results & generate enhanced CSV
   ↓
8. Output: Enhanced CSV + Summary Report
```

### 3.2 MCTS Reasoning Flow (per transaction)
```
Transaction → Initial State
   ↓
MCTS Search Loop (N iterations):
   │
   ├─ Selection: Pick promising node (UCB1)
   │
   ├─ Expansion: LLM generates hypotheses
   │    Example for classification:
   │    - "Business expense - office supplies"
   │    - "Personal purchase - electronics"
   │    - "Recurring subscription - software"
   │
   ├─ Simulation: LLM evaluates each hypothesis
   │    Prompt: "Given transaction X, how likely is hypothesis Y?"
   │    Response: Confidence score + reasoning
   │
   └─ Backpropagation: Update node scores

Best Node → Final Classification/Fraud Decision
```

---

## 4. Technology Stack

### 4.1 Core Dependencies
```toml
[project]
requires-python = ">=3.10,<3.13"
dependencies = [
    "pydantic-ai>=1.12.0",
    "pydantic>=2.0",
    "pandas>=2.0",
    "typer>=0.9"
rich = "^13.0"  # Beautiful CLI output
python-dotenv = "^1.0"  # Environment variable management

# LLM Providers
openai = "^1.0"
anthropic = "^0.18"

# Optional
pyyaml = "^6.0"  # Config file support
pytest = "^7.0"  # Testing
```

### 4.2 Project Structure
```
financial-transaction-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py              # Pydantic AI agent with tools
│   ├── mcts_engine.py        # MCTS reasoning implementation
│   ├── models.py             # Pydantic models
│   ├── config.py             # Configuration management
│   ├── csv_processor.py      # CSV I/O
│   └── cli.py                # CLI interface
├── tests/
│   ├── test_agent.py
│   ├── test_mcts.py
│   ├── test_tools.py
│   └── fixtures/
│       └── sample_transactions.csv
├── examples/
│   ├── sample_input.csv
│   ├── sample_output.csv
│   └── example_config.yaml
├── docs/
│   ├── REQUIREMENTS.md
│   ├── DESIGN.md
│   └── USER_GUIDE.md
├── pyproject.toml
├── uv.lock
├── README.md
└── .env.example
```

---

## 5. Key Design Decisions

### 5.1 Why Pydantic AI?
- Type-safe tool definitions
- Excellent LLM provider abstraction
- Structured outputs via Pydantic models
- Built-in dependency injection for shared state
- Clean async/await support

### 5.2 Why MCTS for Reasoning?
- **Exploration vs Exploitation**: Balances trying new hypotheses vs refining good ones
- **Confidence Calibration**: Visit counts provide natural confidence scores
- **Explainability**: Tree structure shows reasoning path
- **LLM Integration**: LLM acts as simulation policy and hypothesis generator
- **Iterative Refinement**: Multiple iterations improve decision quality

### 5.3 Tool Execution Strategy
- **Sequential**: Tools run in order (1→2→3→4) because each depends on previous
- **Shared State**: Dependencies object passes results between tools
- **Batch Processing**: Classify/fraud detect in batches to reduce LLM calls
- **Error Handling**: Each tool validates inputs before processing

### 5.4 Currency Handling
- Convert all amounts to base currency (GBP) for consistent comparison
- Use fixed exchange rates (configurable) or API-based rates
- Store original currency in output for reference

### 5.5 LLM Provider Abstraction
- Pydantic AI handles provider differences
- Configuration-based model selection
- Reasoning model validation at startup
- Graceful fallback if preferred model unavailable

---

## 6. Security Considerations

### 6.1 API Key Management
- Never hardcode keys
- Support environment variables
- CLI option for key input (not echoed)
- Warn if keys passed as CLI args (visible in process list)

### 6.2 Data Privacy
- No data sent to LLM without user consent
- Option to anonymize transaction descriptions
- No logging of sensitive transaction details
- Secure temporary file handling

### 6.3 Input Validation
- CSV schema validation
- SQL injection prevention (if DB added later)
- Path traversal protection for file I/O
- Reasonable limits on CSV size

---

## 7. Performance Optimization

### 7.1 LLM Call Efficiency
- Batch similar transactions together
- Cache LLM responses for similar queries
- Use streaming for long responses
- Parallel tool calls where possible (not in this sequential design, but future)

### 7.2 MCTS Optimization
- Prune unlikely branches early
- Limit tree depth to prevent explosion
- Use progressive widening
- Stop early if confidence threshold reached

### 7.3 Memory Management
- Stream large CSVs in chunks
- Clear MCTS trees after each transaction
- Use generators where possible
- Limit concurrent LLM calls

---

## 8. Error Handling Strategy

### 8.1 Input Errors
- Invalid CSV format → Clear error message with example
- Missing columns → List missing columns
- Invalid data types → Show problematic rows

### 8.2 LLM Errors
- API rate limits → Exponential backoff + retry
- Invalid API key → Fail fast with helpful message
- Model not found → Suggest alternatives
- Timeout → Configurable timeout with fallback

### 8.3 Processing Errors
- Transaction processing failure → Log and continue with others
- MCTS convergence failure → Use best available result
- Output file write error → Ensure partial results saved

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each tool independently with mock LLM
- Test MCTS engine with deterministic simulations
- Test CSV processor with various formats
- Test configuration validation

### 9.2 Integration Tests
- End-to-end pipeline with sample CSV
- Test with different LLM providers
- Test error scenarios
- Test performance with large CSVs

### 9.3 Validation Tests
- Verify MCTS improves with more iterations
- Verify fraud detection catches known patterns
- Verify classification consistency
- Verify output CSV integrity

---

## 10. Future Enhancements

### 10.1 Performance
- Async parallel processing of transactions
- GPU acceleration for MCTS simulations
- Distributed processing for very large CSVs

### 10.2 Features
- Real-time transaction analysis
- Custom fraud rules DSL
- Machine learning augmentation
- Historical trend analysis
- API server mode

### 10.3 UX
- Web UI
- Progress bar with ETA
- Interactive fraud investigation
- Visualization of MCTS tree
- Export to multiple formats
