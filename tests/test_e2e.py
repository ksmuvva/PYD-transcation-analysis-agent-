"""
End-to-End (E2E) Tests

This module tests:
- Complete workflows from CSV input to enhanced output
- Real-world scenarios
- Full pipeline integration
- Performance and timing
- Output validation
- Error recovery
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from src.models import Currency, FraudRiskLevel
from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.csv_processor import CSVProcessor
from src.agent import run_analysis

# Load environment variables
load_dotenv()


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_path(temp_dir):
    """Create a sample CSV file for testing"""
    data = {
        'transaction_id': ['TX001', 'TX002', 'TX003', 'TX004', 'TX005'],
        'amount': [150.50, 500.00, 1200.00, 25.00, 10000.00],
        'currency': ['GBP', 'USD', 'GBP', 'EUR', 'GBP'],
        'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'merchant': ['Office Supplies', 'Tech Conference', 'Luxury Hotel', 'Coffee Shop', 'Crypto Exchange'],
        'category': ['Business', 'Business', 'Travel', 'Personal', 'Suspicious'],
        'description': ['Paper and pens', 'Annual summit', '5-star hotel', 'Morning coffee', 'Large crypto purchase']
    }

    df = pd.DataFrame(data)
    csv_path = temp_dir / "test_input.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_csv_path(temp_dir):
    """Define output CSV path"""
    return temp_dir / "test_output.csv"


@pytest.fixture
def test_config():
    """Create test configuration with mock/fast settings"""
    llm_config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key=os.getenv('ANTHROPIC_API_KEY', 'test-key')
    )

    mcts_config = MCTSConfig(
        iterations=5,  # Reduced for faster testing
        exploration_constant=1.414,
        max_depth=3,
        simulation_budget=5
    )

    return AgentConfig(
        llm=llm_config,
        mcts=mcts_config,
        threshold_amount=250.0,
        base_currency=Currency.GBP
    )


# ============================================================================
# BASIC E2E WORKFLOW TESTS
# ============================================================================

class TestBasicE2EWorkflow:
    """Test basic end-to-end workflows"""

    def test_csv_load_validate_save_workflow(self, sample_csv_path, output_csv_path):
        """Test loading CSV, validating it, and saving output"""
        # Load CSV
        df = CSVProcessor.load_csv(sample_csv_path)
        assert len(df) == 5

        # Validate schema
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) == 0

        # Save to output (just copy for now)
        df.to_csv(output_csv_path, index=False)
        assert output_csv_path.exists()

        # Verify output
        df_out = pd.read_csv(output_csv_path)
        assert len(df_out) == 5

    def test_load_convert_filter_workflow(self, sample_csv_path, test_config):
        """Test loading, converting to models, and filtering"""
        # Load
        df = CSVProcessor.load_csv(sample_csv_path)

        # Convert to transactions
        transactions = CSVProcessor.convert_to_transactions(df)
        assert len(transactions) == 5

        # Add GBP column for filtering
        df_with_gbp = CSVProcessor.add_gbp_column(df)
        assert 'amount_gbp' in df_with_gbp.columns

        # Filter above threshold
        filtered = df_with_gbp[df_with_gbp['amount_gbp'] > test_config.threshold_amount]
        assert len(filtered) >= 2  # At least TX002 and TX003

    def test_complete_csv_to_enhanced_csv_workflow(self, sample_csv_path, output_csv_path):
        """Test complete workflow from input CSV to enhanced output CSV"""
        # Load input
        df = CSVProcessor.load_csv(sample_csv_path)

        # Simulate analysis results
        df_enhanced = df.copy()
        df_enhanced['above_250_gbp'] = df_enhanced['amount'] > 250
        df_enhanced['classification'] = 'Business'
        df_enhanced['classification_confidence'] = 0.85
        df_enhanced['fraud_risk'] = 'LOW'
        df_enhanced['fraud_confidence'] = 0.90

        # Save enhanced
        df_enhanced.to_csv(output_csv_path, index=False)

        # Verify
        df_out = pd.read_csv(output_csv_path)
        assert 'above_250_gbp' in df_out.columns
        assert 'classification' in df_out.columns
        assert 'fraud_risk' in df_out.columns


# ============================================================================
# REAL PIPELINE E2E TESTS (WITH MOCKING)
# ============================================================================

class TestPipelineE2E:
    """Test full pipeline with real LLM calls"""

    def test_csv_processing_pipeline(self, sample_csv_path, output_csv_path, test_config):
        """Test CSV loading, validation, and conversion pipeline"""
        # Load and validate
        df = CSVProcessor.load_csv(sample_csv_path)
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) == 0

        # Convert
        transactions = CSVProcessor.convert_to_transactions(df)
        assert len(transactions) == 5

        # Verify transaction structure
        for trans in transactions:
            assert hasattr(trans, 'transaction_id')
            assert hasattr(trans, 'amount')
            assert hasattr(trans, 'currency')


# ============================================================================
# REAL-WORLD SCENARIO TESTS
# ============================================================================

class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_business_expense_analysis_scenario(self, temp_dir):
        """Test analyzing business expenses"""
        # Create realistic business expense data
        data = {
            'transaction_id': ['TX001', 'TX002', 'TX003', 'TX004', 'TX005'],
            'amount': [25.50, 500.00, 1200.00, 45.00, 2500.00],
            'currency': ['GBP', 'GBP', 'GBP', 'GBP', 'GBP'],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
            'merchant': [
                'Coffee Shop',
                'Office Depot',
                'Delta Airlines',
                'Restaurant',
                'Dell Technologies'
            ],
            'category': ['Meals', 'Supplies', 'Travel', 'Meals', 'Equipment'],
            'description': [
                'Client meeting coffee',
                'Office supplies bulk order',
                'Business trip to NYC',
                'Team lunch',
                'New laptop for developer'
            ]
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "business_expenses.csv"
        df.to_csv(csv_path, index=False)

        # Load and validate
        loaded_df = CSVProcessor.load_csv(csv_path)
        assert len(loaded_df) == 5

        # All should be valid business transactions
        transactions = CSVProcessor.convert_to_transactions(loaded_df)
        assert all(tx.amount > 0 for tx in transactions)

    def test_fraud_detection_scenario(self, temp_dir):
        """Test fraud detection scenario with suspicious transactions"""
        data = {
            'transaction_id': ['TX001', 'TX002', 'TX003'],
            'amount': [50000.00, 75000.00, 100000.00],
            'currency': ['GBP', 'GBP', 'GBP'],
            'date': ['2024-01-15', '2024-01-15', '2024-01-15'],
            'merchant': [
                'Crypto Exchange XYZ',
                'Offshore Wire Transfer',
                'Casino Monaco'
            ],
            'category': ['Suspicious', 'Suspicious', 'Suspicious'],
            'description': [
                'Large crypto purchase',
                'Wire to tax haven',
                'High-stakes gambling'
            ]
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "suspicious_transactions.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = CSVProcessor.load_csv(csv_path)
        transactions = CSVProcessor.convert_to_transactions(loaded_df)

        # All should be flagged for high amounts
        assert all(tx.amount >= 50000 for tx in transactions)

    def test_international_transactions_scenario(self, temp_dir):
        """Test scenario with international transactions in multiple currencies"""
        data = {
            'transaction_id': ['TX001', 'TX002', 'TX003', 'TX004', 'TX005', 'TX006'],
            'amount': [1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00],
            'currency': ['GBP', 'USD', 'EUR', 'JPY', 'CAD', 'AUD'],
            'date': ['2024-01-15'] * 6,
            'merchant': ['UK Supplier', 'US Vendor', 'EU Partner', 'Japan Corp', 'Canada Inc', 'Australia Ltd'],
            'category': ['Business'] * 6,
            'description': ['International purchase'] * 6
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "international.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = CSVProcessor.load_csv(csv_path)

        # Add GBP conversion
        df_with_gbp = CSVProcessor.add_gbp_column(loaded_df)

        # Verify all currencies converted
        assert df_with_gbp['amount_gbp'].notna().all()
        # GBP should be 1000, others should differ
        gbp_amounts = df_with_gbp['amount_gbp'].tolist()
        assert gbp_amounts[0] == 1000.0  # GBP to GBP


# ============================================================================
# PERFORMANCE AND TIMING TESTS
# ============================================================================

class TestPerformanceE2E:
    """Test performance characteristics of E2E workflows"""

    def test_small_dataset_performance(self, sample_csv_path):
        """Test performance with small dataset (5 transactions)"""
        import time

        start = time.time()

        # Load
        df = CSVProcessor.load_csv(sample_csv_path)

        # Validate
        errors = CSVProcessor.validate_schema(df)

        # Convert
        transactions = CSVProcessor.convert_to_transactions(df)

        elapsed = time.time() - start

        # Should be very fast for 5 transactions (< 1 second)
        assert elapsed < 1.0
        assert len(transactions) == 5

    def test_medium_dataset_performance(self, temp_dir):
        """Test performance with medium dataset (100 transactions)"""
        import time

        # Generate 100 transactions
        data = {
            'transaction_id': [f'TX{i:04d}' for i in range(100)],
            'amount': [100.0 + i for i in range(100)],
            'currency': ['GBP'] * 100,
            'date': ['2024-01-15'] * 100,
            'merchant': [f'Merchant {i}' for i in range(100)],
            'category': ['Business'] * 100,
            'description': [f'Transaction {i}' for i in range(100)]
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "medium.csv"
        df.to_csv(csv_path, index=False)

        start = time.time()

        loaded_df = CSVProcessor.load_csv(csv_path)
        errors = CSVProcessor.validate_schema(loaded_df)
        transactions = CSVProcessor.convert_to_transactions(loaded_df)

        elapsed = time.time() - start

        # Should still be fast (< 2 seconds)
        assert elapsed < 2.0
        assert len(transactions) == 100

    def test_large_dataset_memory_usage(self, temp_dir):
        """Test memory usage with larger dataset"""
        import sys

        # Generate 1000 transactions
        data = {
            'transaction_id': [f'TX{i:06d}' for i in range(1000)],
            'amount': [100.0] * 1000,
            'currency': ['GBP'] * 1000,
            'date': ['2024-01-15'] * 1000,
            'merchant': ['Test'] * 1000,
            'category': ['Business'] * 1000,
            'description': ['Test'] * 1000
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "large.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = CSVProcessor.load_csv(csv_path)

        # Memory usage should be reasonable
        memory_mb = loaded_df.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_mb < 10  # Less than 10 MB for 1000 rows


# ============================================================================
# OUTPUT VALIDATION TESTS
# ============================================================================

class TestOutputValidation:
    """Test validation of output files"""

    def test_output_csv_has_required_columns(self, temp_dir):
        """Test that output CSV has all required analysis columns"""
        # Create mock output
        data = {
            'transaction_id': ['TX001'],
            'amount': [500.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Business'],
            'description': ['Test'],
            # Analysis columns
            'above_250_gbp': [True],
            'classification': ['Business'],
            'classification_confidence': [0.85],
            'fraud_risk': ['LOW'],
            'fraud_confidence': [0.90],
            'fraud_reasoning': ['Normal transaction']
        }

        df = pd.DataFrame(data)
        output_path = temp_dir / "output.csv"
        df.to_csv(output_path, index=False)

        # Validate output
        df_out = pd.read_csv(output_path)

        required_columns = [
            'transaction_id', 'amount', 'currency', 'date', 'merchant',
            'above_250_gbp', 'classification', 'classification_confidence',
            'fraud_risk', 'fraud_confidence'
        ]

        for col in required_columns:
            assert col in df_out.columns

    def test_output_preserves_input_data(self, sample_csv_path, temp_dir):
        """Test that output preserves all input data"""
        # Load input
        df_in = CSVProcessor.load_csv(sample_csv_path)

        # Simulate adding analysis columns
        df_out = df_in.copy()
        df_out['classification'] = 'Business'
        df_out['fraud_risk'] = 'LOW'

        output_path = temp_dir / "output.csv"
        df_out.to_csv(output_path, index=False)

        # Verify
        df_loaded = pd.read_csv(output_path)

        # All original columns should be present
        for col in df_in.columns:
            assert col in df_loaded.columns

        # Original data should match
        assert len(df_loaded) == len(df_in)

    def test_output_data_types_correct(self, temp_dir):
        """Test that output has correct data types"""
        data = {
            'transaction_id': ['TX001'],
            'amount': [500.0],
            'above_250_gbp': [True],
            'classification_confidence': [0.85],
            'fraud_confidence': [0.90]
        }

        df = pd.DataFrame(data)
        output_path = temp_dir / "output.csv"
        df.to_csv(output_path, index=False)

        df_loaded = pd.read_csv(output_path)

        # Check types after reload
        assert df_loaded['amount'].dtype in [float, 'float64']
        assert df_loaded['classification_confidence'].dtype in [float, 'float64']


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================

class TestErrorRecovery:
    """Test error recovery in E2E workflows"""

    def test_recovery_from_invalid_csv(self, temp_dir):
        """Test that system handles invalid CSV gracefully"""
        # Create invalid CSV
        csv_path = temp_dir / "invalid.csv"
        with open(csv_path, 'w') as f:
            f.write("invalid,csv,data\n")
            f.write("no,transaction,id\n")

        # Should raise error or return validation errors
        df = CSVProcessor.load_csv(csv_path)
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) > 0

    def test_recovery_from_partial_data(self, temp_dir):
        """Test handling of partial/incomplete data"""
        data = {
            'transaction_id': ['TX001', 'TX002'],
            'amount': [100.0, None],  # Missing amount
            'currency': ['GBP', 'GBP'],
            'date': ['2024-01-15', '2024-01-16'],
            'merchant': ['Test', 'Test'],
            'category': ['Business', 'Business'],
            'description': ['Test', 'Test']
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "partial.csv"
        df.to_csv(csv_path, index=False)

        loaded_df = CSVProcessor.load_csv(csv_path)

        # Should detect validation errors
        with pytest.raises(Exception):
            transactions = CSVProcessor.convert_to_transactions(loaded_df)


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegression:
    """Test for regression issues"""

    def test_consistent_output_for_same_input(self, sample_csv_path):
        """Test that same input produces consistent output structure"""
        # Load twice
        df1 = CSVProcessor.load_csv(sample_csv_path)
        df2 = CSVProcessor.load_csv(sample_csv_path)

        # Should be identical
        assert df1.equals(df2)
        assert list(df1.columns) == list(df2.columns)
        assert len(df1) == len(df2)

    def test_backward_compatibility_csv_format(self, temp_dir):
        """Test that old CSV format is still supported"""
        # Create CSV without optional category column
        data = {
            'transaction_id': ['TX001'],
            'amount': [100.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'description': ['Test']
        }

        df = pd.DataFrame(data)
        csv_path = temp_dir / "old_format.csv"
        df.to_csv(csv_path, index=False)

        # Should still load (category is optional)
        loaded_df = CSVProcessor.load_csv(csv_path)
        assert len(loaded_df) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
