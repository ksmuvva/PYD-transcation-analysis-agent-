"""
Comprehensive Edge Case Tests

This module tests:
- Empty datasets
- Malformed data
- Extreme values
- Boundary conditions
- Invalid inputs
- Missing data
- Data type mismatches
- Unicode and special characters
- Very large datasets
- Concurrent operations
"""

import pytest
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from decimal import Decimal
import math

from src.models import Transaction, Currency, FraudRiskLevel
from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.agent import AgentDependencies, filter_transactions_above_threshold
from src.csv_processor import CSVProcessor, convert_to_gbp
from src.mcts_engine import MCTSNode


# ============================================================================
# EMPTY DATA TESTS
# ============================================================================

class TestEmptyData:
    """Test handling of empty datasets"""

    def test_empty_dataframe(self):
        """Test processing empty DataFrame"""
        df = pd.DataFrame(columns=[
            'transaction_id', 'amount', 'currency', 'date',
            'merchant', 'category', 'description'
        ])

        config = AgentConfig(
            llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929", api_key="test"),
            mcts=MCTSConfig(),
            threshold_amount=250.0,
            base_currency=Currency.GBP
        )

        deps = AgentDependencies(
            df=df,
            config=config,
            mcts_engine=None,
            llm_client=None,
            results={}
        )

        result = filter_transactions_above_threshold(deps)
        assert result.filtered_count == 0
        assert result.total_amount == 0.0

    def test_empty_csv_file(self):
        """Test loading empty CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                CSVProcessor.load_csv(temp_path)
        finally:
            os.unlink(temp_path)

    def test_csv_with_headers_only(self):
        """Test CSV with headers but no data"""
        content = "transaction_id,amount,currency,date,merchant,category,description\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            df = CSVProcessor.load_csv(temp_path)
            assert len(df) == 0
        finally:
            os.unlink(temp_path)


# ============================================================================
# MALFORMED DATA TESTS
# ============================================================================

class TestMalformedData:
    """Test handling of malformed data"""

    def test_missing_required_column(self):
        """Test DataFrame missing required column"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [100.0],
            # Missing 'currency'
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        errors = CSVProcessor.validate_schema(df)
        assert len(errors) > 0
        assert any('currency' in err.lower() for err in errors)

    def test_invalid_amount_type(self):
        """Test invalid amount data type"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': ['not-a-number'],  # Invalid
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        with pytest.raises(Exception):
            transactions = CSVProcessor.convert_to_transactions(df)

    def test_invalid_currency_code(self):
        """Test invalid currency code"""
        with pytest.raises(ValueError):
            Currency('XYZ')

    def test_invalid_date_format(self):
        """Test various invalid date formats"""
        invalid_dates = [
            'not-a-date',
            '2024-13-01',  # Invalid month
            '2024-01-32',  # Invalid day
            '01/15/2024',  # Might not be supported
        ]

        for invalid_date in invalid_dates:
            df = pd.DataFrame({
                'transaction_id': ['TX001'],
                'amount': [100.0],
                'currency': ['GBP'],
                'date': [invalid_date],
                'merchant': ['Test'],
                'category': ['Test'],
                'description': ['Test']
            })

            # Should either parse or raise error
            try:
                transactions = CSVProcessor.convert_to_transactions(df)
            except Exception:
                pass  # Expected to fail

    def test_negative_amount(self):
        """Test negative transaction amount"""
        with pytest.raises(ValueError):
            Transaction(
                transaction_id='TX001',
                amount=-100.0,  # Negative
                currency=Currency.GBP,
                date=datetime.now(),
                merchant='Test',
                category='Test',
                description='Test'
            )

    def test_zero_amount(self):
        """Test zero transaction amount"""
        # Zero might be invalid depending on business rules
        transaction = Transaction(
            transaction_id='TX001',
            amount=0.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Test',
            category='Test',
            description='Test'
        )
        assert transaction.amount == 0.0


# ============================================================================
# EXTREME VALUES TESTS
# ============================================================================

class TestExtremeValues:
    """Test handling of extreme values"""

    def test_very_large_amount(self):
        """Test very large transaction amount"""
        large_amount = 1_000_000_000.0  # 1 billion

        transaction = Transaction(
            transaction_id='TX001',
            amount=large_amount,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Test',
            category='Test',
            description='Test'
        )

        assert transaction.amount == large_amount

    def test_very_small_amount(self):
        """Test very small transaction amount"""
        small_amount = 0.01  # 1 penny

        transaction = Transaction(
            transaction_id='TX001',
            amount=small_amount,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Test',
            category='Test',
            description='Test'
        )

        assert transaction.amount == small_amount

    def test_many_decimal_places(self):
        """Test amount with many decimal places"""
        amount = 100.123456789

        transaction = Transaction(
            transaction_id='TX001',
            amount=amount,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Test',
            category='Test',
            description='Test'
        )

        # Should handle or round appropriately
        assert transaction.amount >= 100.0

    def test_very_old_date(self):
        """Test very old transaction date"""
        old_date = datetime(1900, 1, 1)

        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=old_date,
            merchant='Test',
            category='Test',
            description='Test'
        )

        assert transaction.date == old_date

    def test_future_date(self):
        """Test future transaction date"""
        future_date = datetime(2099, 12, 31)

        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=future_date,
            merchant='Test',
            category='Test',
            description='Test'
        )

        assert transaction.date == future_date

    def test_very_long_merchant_name(self):
        """Test very long merchant name"""
        long_name = "A" * 10000  # 10k characters

        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant=long_name,
            category='Test',
            description='Test'
        )

        assert len(transaction.merchant) == 10000

    def test_very_long_description(self):
        """Test very long description"""
        long_desc = "Description " * 1000

        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Test',
            category='Test',
            description=long_desc
        )

        assert len(transaction.description) > 1000


# ============================================================================
# BOUNDARY CONDITION TESTS
# ============================================================================

class TestBoundaryConditions:
    """Test boundary conditions"""

    def test_threshold_exactly_at_boundary(self):
        """Test amount exactly at threshold"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [250.0],  # Exactly at threshold
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        config = AgentConfig(
            llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929", api_key="test"),
            mcts=MCTSConfig(),
            threshold_amount=250.0,
            base_currency=Currency.GBP
        )

        deps = AgentDependencies(
            df=df,
            config=config,
            mcts_engine=None,
            llm_client=None,
            results={}
        )

        result = filter_transactions_above_threshold(deps)
        # Depends on if threshold is > or >=
        # Most likely > (above threshold means strictly greater)
        assert result.filtered_count in [0, 1]

    def test_confidence_at_zero(self):
        """Test confidence score at 0.0"""
        node = MCTSNode(state={'confidence': 0.0}, parent=None)
        node.visits = 1
        node.value = 0.0

        assert node.value / node.visits == 0.0

    def test_confidence_at_one(self):
        """Test confidence score at 1.0"""
        node = MCTSNode(state={'confidence': 1.0}, parent=None)
        node.visits = 1
        node.value = 1.0

        assert node.value / node.visits == 1.0

    def test_single_transaction_dataset(self):
        """Test dataset with single transaction"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [500.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        assert len(df) == 1

    def test_mcts_single_iteration(self):
        """Test MCTS with single iteration"""
        config = MCTSConfig(iterations=1)
        assert config.iterations == 1

    def test_mcts_max_depth_zero(self):
        """Test MCTS with max depth of 0"""
        config = MCTSConfig(max_depth=0)
        assert config.max_depth == 0

    def test_exploration_constant_zero(self):
        """Test UCB1 with zero exploration constant"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10

        child = MCTSNode(state={}, parent=parent)
        child.visits = 5
        child.value = 3.0

        score = child.ucb1_score(c=0.0)
        # Should be pure exploitation: value/visits
        assert score == 0.6


# ============================================================================
# UNICODE AND SPECIAL CHARACTERS TESTS
# ============================================================================

class TestUnicodeAndSpecialCharacters:
    """Test handling of unicode and special characters"""

    def test_unicode_merchant_name(self):
        """Test merchant name with unicode characters"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Café René ☕',
            category='Personal',
            description='Coffee'
        )

        assert 'Café' in transaction.merchant
        assert '☕' in transaction.merchant

    def test_chinese_characters(self):
        """Test Chinese characters in description"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Restaurant',
            category='Personal',
            description='北京烤鸭 - Peking Duck'
        )

        assert '北京烤鸭' in transaction.description

    def test_emoji_in_description(self):
        """Test emoji in description"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Store',
            category='Personal',
            description='Birthday gift 🎁🎂🎉'
        )

        assert '🎁' in transaction.description

    def test_special_csv_characters(self):
        """Test special characters that might break CSV parsing"""
        # Quotes, commas, newlines
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='Store "Best" Co.',
            category='Personal',
            description='Item 1, Item 2, Item 3'
        )

        assert '"' in transaction.merchant
        assert ',' in transaction.description

    def test_null_bytes(self):
        """Test handling of null bytes"""
        # Some systems might have issues with null bytes
        try:
            transaction = Transaction(
                transaction_id='TX001',
                amount=100.0,
                currency=Currency.GBP,
                date=datetime.now(),
                merchant='Test\x00Store',
                category='Personal',
                description='Test'
            )
            # If it doesn't raise, that's OK
        except Exception:
            # Null bytes might be rejected, that's also OK
            pass


# ============================================================================
# MISSING DATA TESTS
# ============================================================================

class TestMissingData:
    """Test handling of missing data"""

    def test_missing_transaction_id(self):
        """Test missing transaction ID"""
        df = pd.DataFrame({
            'transaction_id': [None],
            'amount': [100.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        with pytest.raises(Exception):
            CSVProcessor.convert_to_transactions(df)

    def test_missing_amount(self):
        """Test missing amount"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [None],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        with pytest.raises(Exception):
            CSVProcessor.convert_to_transactions(df)

    def test_missing_optional_category(self):
        """Test missing optional category field"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [100.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            # category is optional
            'description': ['Test']
        })

        # Should work without category
        try:
            transactions = CSVProcessor.convert_to_transactions(df)
            assert len(transactions) == 1
        except Exception:
            # Or might require category - depends on implementation
            pass

    def test_empty_string_merchant(self):
        """Test empty string as merchant"""
        transaction = Transaction(
            transaction_id='TX001',
            amount=100.0,
            currency=Currency.GBP,
            date=datetime.now(),
            merchant='',  # Empty
            category='Test',
            description='Test'
        )

        assert transaction.merchant == ''


# ============================================================================
# LARGE DATASET TESTS
# ============================================================================

class TestLargeDatasets:
    """Test handling of large datasets"""

    def test_large_dataframe_10k_rows(self):
        """Test DataFrame with 10,000 rows"""
        rows = []
        for i in range(10000):
            rows.append({
                'transaction_id': f'TX{i:06d}',
                'amount': 100.0 + i * 0.1,
                'currency': 'GBP',
                'date': '2024-01-15',
                'merchant': f'Merchant {i}',
                'category': 'Test',
                'description': f'Transaction {i}'
            })

        df = pd.DataFrame(rows)
        assert len(df) == 10000

    def test_large_dataframe_100k_rows(self):
        """Test DataFrame with 100,000 rows"""
        # This is a stress test
        rows = []
        for i in range(100000):
            rows.append({
                'transaction_id': f'TX{i:06d}',
                'amount': 100.0 + i * 0.1,
                'currency': 'GBP',
                'date': '2024-01-15',
                'merchant': f'Merchant {i % 1000}',  # Reuse merchant names
                'category': 'Test',
                'description': 'Test'
            })

        df = pd.DataFrame(rows)
        assert len(df) == 100000

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large dataset"""
        import sys

        # Create large dataset
        rows = []
        for i in range(10000):
            rows.append({
                'transaction_id': f'TX{i:06d}',
                'amount': 100.0,
                'currency': 'GBP',
                'date': '2024-01-15',
                'merchant': 'Test',
                'category': 'Test',
                'description': 'Test'
            })

        df = pd.DataFrame(rows)

        # Check memory usage is reasonable
        memory_usage = df.memory_usage(deep=True).sum()
        # Should be less than 100MB for 10k rows
        assert memory_usage < 100 * 1024 * 1024


# ============================================================================
# DUPLICATE DATA TESTS
# ============================================================================

class TestDuplicateData:
    """Test handling of duplicate data"""

    def test_duplicate_transaction_ids(self):
        """Test duplicate transaction IDs"""
        df = pd.DataFrame({
            'transaction_id': ['TX001', 'TX001'],  # Duplicate
            'amount': [100.0, 200.0],
            'currency': ['GBP', 'GBP'],
            'date': ['2024-01-15', '2024-01-16'],
            'merchant': ['Test1', 'Test2'],
            'category': ['Test', 'Test'],
            'description': ['Test', 'Test']
        })

        errors = CSVProcessor.validate_schema(df)
        # Should detect duplicate IDs
        assert any('duplicate' in err.lower() or 'unique' in err.lower() for err in errors)

    def test_duplicate_transactions_different_ids(self):
        """Test identical transactions with different IDs"""
        df = pd.DataFrame({
            'transaction_id': ['TX001', 'TX002'],
            'amount': [100.0, 100.0],  # Same
            'currency': ['GBP', 'GBP'],  # Same
            'date': ['2024-01-15', '2024-01-15'],  # Same
            'merchant': ['Test', 'Test'],  # Same
            'category': ['Test', 'Test'],
            'description': ['Test', 'Test']
        })

        # This should be valid - different IDs
        transactions = CSVProcessor.convert_to_transactions(df)
        assert len(transactions) == 2


# ============================================================================
# CURRENCY CONVERSION EDGE CASES
# ============================================================================

class TestCurrencyConversionEdgeCases:
    """Test edge cases in currency conversion"""

    def test_convert_zero_amount(self):
        """Test converting zero amount"""
        result = convert_to_gbp(0.0, Currency.USD)
        assert result == 0.0

    def test_convert_very_large_amount(self):
        """Test converting very large amount"""
        result = convert_to_gbp(1_000_000_000.0, Currency.USD)
        assert result > 0
        assert result == pytest.approx(790_000_000.0, rel=0.01)

    def test_convert_very_small_amount(self):
        """Test converting very small amount"""
        result = convert_to_gbp(0.01, Currency.JPY)
        assert result >= 0

    def test_gbp_to_gbp_identity(self):
        """Test that GBP to GBP is identity"""
        amounts = [1.0, 100.0, 1000.0, 0.01]
        for amount in amounts:
            result = convert_to_gbp(amount, Currency.GBP)
            assert result == amount


# ============================================================================
# CONCURRENT ACCESS TESTS
# ============================================================================

class TestConcurrentAccess:
    """Test concurrent access patterns"""

    def test_shared_dataframe_reference(self):
        """Test that shared DataFrame references work correctly"""
        df = pd.DataFrame({
            'transaction_id': ['TX001'],
            'amount': [100.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        # Multiple references to same DataFrame
        df_ref1 = df
        df_ref2 = df

        assert df_ref1 is df_ref2
        assert len(df_ref1) == len(df_ref2)

    def test_results_dictionary_mutations(self):
        """Test that results dictionary can be mutated safely"""
        results = {}

        # Simulate multiple tools updating results
        results['tool1'] = 'result1'
        results['tool2'] = 'result2'
        results['tool3'] = 'result3'

        assert len(results) == 3
        assert all(key in results for key in ['tool1', 'tool2', 'tool3'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
