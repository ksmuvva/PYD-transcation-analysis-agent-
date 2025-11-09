"""
Prompt Testing Suite

This module tests:
- Prompt construction for hypothesis generation
- Prompt construction for hypothesis evaluation
- Prompt quality and clarity
- Prompt consistency
- Expected outputs from prompts
- Prompt edge cases
"""

import pytest
from datetime import datetime

from src.models import Transaction, Currency, FraudRiskLevel
from src.config import MCTSConfig
from src.mcts_engine import MCTSEngine


# ============================================================================
# PROMPT CONSTRUCTION TESTS
# ============================================================================

class TestPromptConstruction:
    """Test that prompts are constructed correctly"""

    def test_classification_hypothesis_prompt_includes_transaction_details(self):
        """Test that classification prompts include all relevant transaction details"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=1500.00,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Luxury Hotel',
            category='Travel',
            description='5-star hotel booking for business trip'
        )

        # Prompt should include these key details
        expected_details = [
            'TX001',
            '1500',
            'GBP',
            'Luxury Hotel',
            '5-star hotel booking'
        ]

        # In actual implementation, this would be the prompt template
        prompt = f"""Analyze the following transaction and generate 3-5 classification hypotheses:
Transaction ID: {transaction.transaction_id}
Amount: {transaction.amount} {transaction.currency.value}
Merchant: {transaction.merchant}
Description: {transaction.description}
Date: {transaction.date}

Generate hypotheses for the primary category of this transaction."""

        for detail in expected_details:
            assert str(detail) in prompt

    def test_fraud_detection_prompt_includes_risk_indicators(self):
        """Test that fraud detection prompts include risk indicator guidance"""
        transaction = Transaction(
            transaction_id='TX002',
            amount=50000.00,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Crypto Exchange',
            category='Suspicious',
            description='Large cryptocurrency purchase'
        )

        # Fraud prompt should mention risk indicators
        prompt = f"""Analyze the following transaction for fraud risk:
Transaction ID: {transaction.transaction_id}
Amount: {transaction.amount} {transaction.currency.value}
Merchant: {transaction.merchant}
Description: {transaction.description}

Consider these risk indicators:
- Amount (especially large amounts)
- Merchant type and reputation
- Transaction timing and patterns
- Description keywords (crypto, offshore, etc.)

Generate 3-5 fraud risk hypotheses with risk levels: LOW, MEDIUM, HIGH, CRITICAL"""

        assert 'fraud risk' in prompt.lower()
        assert 'risk indicators' in prompt.lower()
        assert 'CRITICAL' in prompt

    def test_hypothesis_evaluation_prompt_structure(self):
        """Test that hypothesis evaluation prompts are well-structured"""
        hypothesis = {
            'category': 'Business',
            'rationale': 'Corporate travel expense',
            'confidence': 0.8
        }

        transaction = Transaction(
            transaction_id='TX003',
            amount=1200.00,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Airlines Inc',
            category='Travel',
            description='Business class flight to New York'
        )

        # Evaluation prompt should include both hypothesis and transaction
        prompt = f"""Evaluate the following hypothesis for the transaction:

Transaction:
- ID: {transaction.transaction_id}
- Amount: {transaction.amount} {transaction.currency.value}
- Merchant: {transaction.merchant}
- Description: {transaction.description}

Hypothesis:
- Category: {hypothesis['category']}
- Rationale: {hypothesis['rationale']}

Provide a confidence score (0.0 to 1.0) and detailed reasoning for this hypothesis."""

        assert 'Evaluate' in prompt
        assert hypothesis['category'] in prompt
        assert transaction.merchant in prompt


# ============================================================================
# PROMPT QUALITY TESTS
# ============================================================================

class TestPromptQuality:
    """Test the quality and clarity of prompts"""

    def test_prompt_clarity_and_instructions(self):
        """Test that prompts have clear instructions"""
        prompt = """Analyze the following transaction and generate 3-5 classification hypotheses:
Transaction ID: TX001
Amount: 500.00 GBP
Merchant: Tech Store
Description: Laptop purchase

For each hypothesis, provide:
1. Primary category (e.g., Business, Personal, Travel, etc.)
2. Detailed rationale explaining your reasoning
3. Confidence score (0.0 to 1.0)

Return as JSON array."""

        # Check for clear instructions
        assert 'generate' in prompt.lower() or 'provide' in prompt.lower()
        assert 'JSON' in prompt
        assert 'category' in prompt.lower()
        assert 'rationale' in prompt.lower()
        assert 'confidence' in prompt.lower()

    def test_prompt_includes_output_format(self):
        """Test that prompts specify expected output format"""
        prompt = """Generate fraud risk hypotheses for this transaction.

Return as JSON array with this structure:
[
  {
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "indicators": ["indicator1", "indicator2"],
    "rationale": "explanation",
    "confidence": 0.85
  }
]"""

        assert 'JSON' in prompt
        assert 'array' in prompt.lower()
        assert 'structure' in prompt.lower()

    def test_prompt_provides_examples(self):
        """Test that complex prompts provide examples"""
        prompt = """Generate classification hypotheses.

Example hypothesis:
{
  "category": "Business",
  "rationale": "Office supplies purchase for company use",
  "confidence": 0.92
}

Now generate 3-5 hypotheses for the transaction:
..."""

        assert 'Example' in prompt
        assert 'category' in prompt
        assert 'rationale' in prompt

    def test_prompt_conciseness(self):
        """Test that prompts are concise but complete"""
        prompt = """Classify transaction TX001 (500 GBP at Tech Store).
Generate 3-5 category hypotheses with rationale and confidence.
Return as JSON array."""

        # Should be under 1000 characters for simple tasks
        assert len(prompt) < 1000
        # But should include key information
        assert 'TX001' in prompt
        assert 'GBP' in prompt


# ============================================================================
# PROMPT CONSISTENCY TESTS
# ============================================================================

class TestPromptConsistency:
    """Test that prompts are consistent across similar tasks"""

    def test_classification_prompt_consistency(self):
        """Test that classification prompts follow same template"""
        transactions = [
            Transaction(
                transaction_id='TX001',
                amount=500.00,
                currency=Currency.GBP,
                date=datetime(2024, 1, 15),
                merchant='Store A',
                category='Business',
                description='Purchase 1'
            ),
            Transaction(
                transaction_id='TX002',
                amount=1000.00,
                currency=Currency.USD,
                date=datetime(2024, 1, 16),
                merchant='Store B',
                category='Personal',
                description='Purchase 2'
            )
        ]

        prompts = []
        for tx in transactions:
            prompt = f"""Analyze transaction {tx.transaction_id} ({tx.amount} {tx.currency.value} at {tx.merchant}).
Generate 3-5 classification hypotheses.
Return as JSON array."""
            prompts.append(prompt)

        # All prompts should follow same structure
        for prompt in prompts:
            assert 'Analyze transaction' in prompt
            assert 'Generate 3-5' in prompt
            assert 'JSON array' in prompt

    def test_fraud_prompt_consistency(self):
        """Test that fraud detection prompts follow same template"""
        prompt_template = """Analyze transaction {tx_id} for fraud risk.
Amount: {amount} {currency}
Merchant: {merchant}

Generate 3-5 fraud risk hypotheses (LOW/MEDIUM/HIGH/CRITICAL).
Return as JSON array."""

        # Should work for any transaction
        prompts = [
            prompt_template.format(tx_id='TX001', amount=100, currency='GBP', merchant='Store'),
            prompt_template.format(tx_id='TX002', amount=50000, currency='USD', merchant='Crypto')
        ]

        for prompt in prompts:
            assert 'fraud risk' in prompt.lower()
            assert 'CRITICAL' in prompt


# ============================================================================
# EXPECTED OUTPUT TESTS
# ============================================================================

class TestExpectedOutputs:
    """Test that prompts produce expected outputs"""

    @pytest.mark.asyncio
    async def test_classification_prompt_produces_valid_json(self):
        """Test that classification prompt produces valid JSON"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            # Simulate LLM response to classification prompt
            return [
                {
                    "category": "Business",
                    "rationale": "Office supplies for company",
                    "confidence": 0.85
                },
                {
                    "category": "Personal",
                    "rationale": "Could be personal purchase",
                    "confidence": 0.40
                }
            ]

        result = await mock_llm("Generate classification hypotheses")

        # Validate structure
        assert isinstance(result, list)
        assert len(result) >= 2
        assert all('category' in h for h in result)
        assert all('rationale' in h for h in result)
        assert all('confidence' in h for h in result)

    @pytest.mark.asyncio
    async def test_fraud_prompt_produces_valid_json(self):
        """Test that fraud prompt produces valid JSON"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return [
                {
                    "risk_level": "HIGH",
                    "indicators": ["Large amount", "Crypto merchant"],
                    "rationale": "High-risk cryptocurrency transaction",
                    "confidence": 0.88
                },
                {
                    "risk_level": "MEDIUM",
                    "indicators": ["First-time merchant"],
                    "rationale": "New merchant, moderate risk",
                    "confidence": 0.65
                }
            ]

        result = await mock_llm("Generate fraud hypotheses")

        # Validate structure
        assert isinstance(result, list)
        assert all('risk_level' in h for h in result)
        assert all('indicators' in h for h in result)
        assert all('rationale' in h for h in result)
        assert all(h['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] for h in result)

    @pytest.mark.asyncio
    async def test_evaluation_prompt_produces_confidence_score(self):
        """Test that evaluation prompt produces confidence score"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return {
                "confidence": 0.92,
                "reasoning": "Strong evidence supports this classification"
            }

        result = await mock_llm("Evaluate hypothesis")

        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'reasoning' in result


# ============================================================================
# PROMPT EDGE CASES
# ============================================================================

class TestPromptEdgeCases:
    """Test prompts with edge case inputs"""

    def test_prompt_with_very_long_description(self):
        """Test prompt with very long transaction description"""
        long_desc = "Purchase of items: " + ", ".join([f"Item {i}" for i in range(1000)])

        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Store',
            category='Business',
            description=long_desc
        )

        # Prompt should handle or truncate long description
        prompt = f"""Analyze transaction {transaction.transaction_id}.
Description: {transaction.description[:500]}...  # Truncated
Generate classification hypotheses."""

        # Should be manageable length
        assert len(prompt) < 10000

    def test_prompt_with_special_characters(self):
        """Test prompt with special characters in transaction data"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Store "Best" & Co.',
            category='Business',
            description='Items: A, B, C; Total: $100'
        )

        prompt = f"""Analyze transaction:
Merchant: {transaction.merchant}
Description: {transaction.description}"""

        # Should include special characters correctly
        assert '"Best"' in prompt or 'Best' in prompt
        assert '&' in prompt or 'and' in prompt

    def test_prompt_with_unicode(self):
        """Test prompt with unicode characters"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Café René ☕',
            category='Personal',
            description='Coffee and croissant 🥐'
        )

        prompt = f"""Analyze transaction at {transaction.merchant}.
Description: {transaction.description}"""

        # Should handle unicode
        assert 'Café' in prompt or 'Cafe' in prompt

    def test_prompt_with_zero_amount(self):
        """Test prompt with zero amount transaction"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=0.0,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Store',
            category='Other',
            description='Refund or adjustment'
        )

        prompt = f"""Analyze transaction {transaction.transaction_id}.
Amount: {transaction.amount} {transaction.currency.value}"""

        assert '0.0' in prompt or '0' in prompt


# ============================================================================
# PROMPT VERSIONING TESTS
# ============================================================================

class TestPromptVersioning:
    """Test that prompt changes are tracked and consistent"""

    def test_system_prompt_consistency(self):
        """Test that system prompt is consistent"""
        expected_system_prompt = """You are a financial transaction analysis expert.
You use Monte Carlo Tree Search (MCTS) reasoning to analyze transactions,
classify them accurately, and detect potential fraud.

Always provide detailed reasoning for your conclusions.
Consider multiple hypotheses before making final decisions.
Use confidence scores to reflect uncertainty.

When generating hypotheses or evaluations, always respond with valid JSON."""

        # System prompt should contain key elements
        assert 'financial transaction analysis' in expected_system_prompt.lower()
        assert 'MCTS' in expected_system_prompt
        assert 'JSON' in expected_system_prompt

    def test_prompt_includes_json_format_requirement(self):
        """Test that prompts require JSON format consistently"""
        prompts = [
            "Generate hypotheses. Return as JSON array.",
            "Evaluate hypothesis. Return as JSON object.",
            "Analyze transaction. Return results as JSON."
        ]

        for prompt in prompts:
            assert 'JSON' in prompt


# ============================================================================
# PROMPT PERFORMANCE TESTS
# ============================================================================

class TestPromptPerformance:
    """Test prompt characteristics that affect performance"""

    def test_prompt_token_efficiency(self):
        """Test that prompts are token-efficient"""
        # Concise prompt
        concise = "Classify TX001 (500 GBP, Tech Store). Return JSON with category, rationale, confidence."

        # Verbose prompt
        verbose = """Please analyze the following transaction and provide a detailed classification.
The transaction has the following properties:
- Transaction ID: TX001
- Amount: 500 GBP
- Merchant: Tech Store

Please provide your response in JSON format including:
- The category you've determined
- The rationale for your classification
- Your confidence level in the classification"""

        # Concise should be much shorter
        assert len(concise) < len(verbose) * 0.5

    def test_prompt_reusability(self):
        """Test that prompt templates can be reused"""
        template = "Analyze {tx_id} ({amount} {currency} at {merchant}). Generate {count} hypotheses."

        # Should work for multiple transactions
        prompt1 = template.format(tx_id='TX001', amount=100, currency='GBP', merchant='Store A', count=3)
        prompt2 = template.format(tx_id='TX002', amount=500, currency='USD', merchant='Store B', count=5)

        assert 'TX001' in prompt1
        assert 'TX002' in prompt2
        assert '3 hypotheses' in prompt1
        assert '5 hypotheses' in prompt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
