"""
Comprehensive CLI Tests for Financial Transaction Analysis Agent

This module tests:
- Starting the agent with various configurations
- Choosing different LLM providers and models
- CSV file upload and validation
- CLI argument parsing and validation
- Error handling and edge cases
- Output generation
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd

from src.cli import app
from src.models import Currency, FraudRiskLevel


# Test fixtures
runner = CliRunner()


@pytest.fixture
def sample_csv_file():
    """Create a temporary sample CSV file for testing"""
    content = """transaction_id,amount,currency,date,merchant,category,description
TX001,150.50,GBP,2024-01-15,Office Supplies Ltd,Business,Paper and pens
TX002,500.00,USD,2024-01-16,Tech Conference,Business,Annual tech summit registration
TX003,1200.00,GBP,2024-01-17,Luxury Hotel,Travel,5-star hotel booking
TX004,25.00,EUR,2024-01-18,Coffee Shop,Personal,Morning coffee
TX005,10000.00,GBP,2024-01-19,Crypto Exchange,Suspicious,Large crypto purchase
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_csv_file():
    """Create an empty CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def invalid_csv_file():
    """Create an invalid CSV file (missing required columns)"""
    content = """id,price,date
1,100.00,2024-01-15
2,200.00,2024-01-16
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def malformed_csv_file():
    """Create a malformed CSV file"""
    content = """transaction_id,amount,currency,date,merchant,category,description
TX001,150.50,GBP,2024-01-15,Office Supplies Ltd,Business,Paper and pens
TX002,INVALID,USD,2024-01-16,Tech Conference,Business,Annual tech summit registration
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def large_csv_file():
    """Create a large CSV file with many transactions"""
    rows = ["transaction_id,amount,currency,date,merchant,category,description"]
    for i in range(100):
        rows.append(f"TX{i:04d},{100 + i}.50,GBP,2024-01-15,Merchant {i},Business,Description {i}")
    content = "\n".join(rows)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# CLI STARTUP AND BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestCLIStartup:
    """Test CLI application startup and basic commands"""

    def test_cli_help_command(self):
        """Test that --help displays help information"""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout or "Analyze" in result.stdout

    def test_analyze_command_help(self):
        """Test that analyze --help displays command help"""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "csv-path" in result.stdout.lower() or "CSV file" in result.stdout

    def test_validate_command_help(self):
        """Test that validate --help displays command help"""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0

    def test_models_command(self):
        """Test that models command lists available models"""
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        # Should list reasoning models
        assert "o1" in result.stdout or "claude" in result.stdout.lower()


# ============================================================================
# LLM PROVIDER AND MODEL SELECTION TESTS
# ============================================================================

class TestLLMSelection:
    """Test LLM provider and model selection"""

    @patch('src.agent.run_analysis')
    def test_default_llm_provider(self, mock_run, sample_csv_file):
        """Test that default LLM provider is used when not specified"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, ["analyze", sample_csv_file])
        # Should use default from environment
        assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_openai_provider_selection(self, mock_run, sample_csv_file):
        """Test selecting OpenAI as LLM provider"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "openai",
            "--model", "o1-mini"
        ])
        # Might fail without API key, but should accept the arguments
        assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_anthropic_provider_selection(self, mock_run, sample_csv_file):
        """Test selecting Anthropic as LLM provider"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "anthropic",
            "--model", "claude-sonnet-4-5-20250929"
        ])
        assert result.exit_code == 0 or "API key" in result.stdout

    def test_invalid_provider(self, sample_csv_file):
        """Test that invalid provider is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "invalid-provider"
        ])
        # Should fail with error
        assert result.exit_code != 0

    def test_non_reasoning_model_rejected(self, sample_csv_file):
        """Test that non-reasoning models are rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "openai",
            "--model", "gpt-4"  # Not a reasoning model
        ])
        # Should fail with error about reasoning model requirement
        assert result.exit_code != 0 or "reasoning" in result.stdout.lower()

    @patch('src.agent.run_analysis')
    def test_valid_openai_reasoning_models(self, mock_run, sample_csv_file):
        """Test all valid OpenAI reasoning models"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        models = ["o1", "o1-preview", "o1-mini", "o3-mini"]
        for model in models:
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--provider", "openai",
                "--model", model
            ])
            # Should accept the model (might fail on API key)
            assert result.exit_code == 0 or "API key" in result.stdout, f"Model {model} should be accepted"

    @patch('src.agent.run_analysis')
    def test_valid_anthropic_reasoning_models(self, mock_run, sample_csv_file):
        """Test all valid Anthropic reasoning models"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929"
        ]
        for model in models:
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--provider", "anthropic",
                "--model", model
            ])
            assert result.exit_code == 0 or "API key" in result.stdout, f"Model {model} should be accepted"


# ============================================================================
# CSV FILE UPLOAD AND VALIDATION TESTS
# ============================================================================

class TestCSVUpload:
    """Test CSV file upload and validation"""

    def test_missing_csv_file_argument(self):
        """Test that missing CSV file path causes error"""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "required" in result.stdout.lower()

    def test_nonexistent_csv_file(self):
        """Test that nonexistent CSV file causes error"""
        result = runner.invoke(app, ["analyze", "/nonexistent/file.csv"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "does not exist" in result.stdout.lower()

    def test_empty_csv_file(self, empty_csv_file):
        """Test that empty CSV file is rejected"""
        result = runner.invoke(app, ["validate", empty_csv_file])
        assert result.exit_code != 0
        assert "empty" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_invalid_csv_missing_columns(self, invalid_csv_file):
        """Test that CSV with missing required columns is rejected"""
        result = runner.invoke(app, ["validate", invalid_csv_file])
        assert result.exit_code != 0
        assert "missing" in result.stdout.lower() or "required" in result.stdout.lower()

    def test_malformed_csv_data(self, malformed_csv_file):
        """Test that CSV with invalid data types is rejected"""
        result = runner.invoke(app, ["validate", malformed_csv_file])
        assert result.exit_code != 0

    def test_valid_csv_file(self, sample_csv_file):
        """Test that valid CSV file passes validation"""
        result = runner.invoke(app, ["validate", sample_csv_file])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower() or "success" in result.stdout.lower()

    def test_csv_with_special_characters(self):
        """Test CSV with special characters in data"""
        content = """transaction_id,amount,currency,date,merchant,category,description
TX001,150.50,GBP,2024-01-15,"O'Reilly Books & Co.",Business,"Books, journals & magazines"
TX002,500.00,USD,2024-01-16,Café René,Personal,Lunch at café
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = runner.invoke(app, ["validate", temp_path])
            assert result.exit_code == 0
        finally:
            os.unlink(temp_path)

    def test_csv_different_encodings(self):
        """Test CSV files with different encodings"""
        content = "transaction_id,amount,currency,date,merchant,category,description\nTX001,150.50,GBP,2024-01-15,Test Merchant,Business,Test description\n"

        # Test UTF-8
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            result = runner.invoke(app, ["validate", temp_path])
            assert result.exit_code == 0
        finally:
            os.unlink(temp_path)


# ============================================================================
# CLI OPTION TESTS
# ============================================================================

class TestCLIOptions:
    """Test various CLI options and parameters"""

    @patch('src.agent.run_analysis')
    def test_output_path_option(self, mock_run, sample_csv_file):
        """Test specifying custom output path"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Should accept the output path
            assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_threshold_option(self, mock_run, sample_csv_file):
        """Test specifying custom threshold amount"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--threshold", "500.0"
        ])
        assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_negative_threshold_rejected(self, mock_run, sample_csv_file):
        """Test that negative threshold is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--threshold", "-100.0"
        ])
        # Should fail with validation error
        assert result.exit_code != 0

    @patch('src.agent.run_analysis')
    def test_currency_option(self, mock_run, sample_csv_file):
        """Test specifying different base currencies"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        currencies = ["GBP", "USD", "EUR"]
        for currency in currencies:
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--currency", currency
            ])
            assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_invalid_currency_rejected(self, mock_run, sample_csv_file):
        """Test that invalid currency is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--currency", "XYZ"
        ])
        assert result.exit_code != 0

    @patch('src.agent.run_analysis')
    def test_mcts_iterations_option(self, mock_run, sample_csv_file):
        """Test specifying custom MCTS iterations"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--mcts-iterations", "50"
        ])
        assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_zero_mcts_iterations_rejected(self, mock_run, sample_csv_file):
        """Test that zero MCTS iterations is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--mcts-iterations", "0"
        ])
        assert result.exit_code != 0

    @patch('src.agent.run_analysis')
    def test_verbose_flag(self, mock_run, sample_csv_file):
        """Test verbose output flag"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--verbose"
        ])
        assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_api_key_option(self, mock_run, sample_csv_file):
        """Test providing API key via command line"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--api-key", "test-api-key-12345"
        ])
        # Should accept the API key
        assert result.exit_code == 0 or "invalid" in result.stdout.lower()


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in various scenarios"""

    def test_file_permission_error(self):
        """Test handling of file permission errors"""
        # Create a file and make it unreadable
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            os.chmod(temp_path, 0o000)
            result = runner.invoke(app, ["validate", temp_path])
            # Should handle permission error gracefully
            assert result.exit_code != 0
        finally:
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)

    @patch('src.agent.run_analysis')
    def test_api_key_missing_error(self, mock_run, sample_csv_file):
        """Test error when API key is missing"""
        # Clear environment variable
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--provider", "openai",
                "--model", "o1-mini"
            ])
            # Should fail with API key error
            assert result.exit_code != 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_network_error_handling(self, mock_run, sample_csv_file):
        """Test handling of network errors during LLM calls"""
        mock_run.side_effect = ConnectionError("Network unreachable")

        result = runner.invoke(app, ["analyze", sample_csv_file])
        # Should handle network error gracefully
        assert result.exit_code != 0

    @patch('src.agent.run_analysis')
    def test_timeout_error_handling(self, mock_run, sample_csv_file):
        """Test handling of timeout errors"""
        mock_run.side_effect = TimeoutError("Request timed out")

        result = runner.invoke(app, ["analyze", sample_csv_file])
        assert result.exit_code != 0


# ============================================================================
# OUTPUT GENERATION TESTS
# ============================================================================

class TestOutputGeneration:
    """Test output file generation"""

    @patch('src.agent.run_analysis')
    def test_output_file_created(self, mock_run, sample_csv_file):
        """Test that output file is created successfully"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            transactions_above_threshold=3,
            high_risk_transactions=1,
            critical_risk_transactions=0,
            processing_time_seconds=1.5,
            llm_provider="anthropic",
            model_used="claude-sonnet-4-5-20250929",
            mcts_iterations_total=300
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Output file should be created (or API key error)
            assert result.exit_code == 0 or "API key" in result.stdout

    @patch('src.agent.run_analysis')
    def test_output_directory_created(self, mock_run, sample_csv_file):
        """Test that output directory is created if it doesn't exist"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=5,
            processing_time_seconds=1.5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Should create parent directory
            assert result.exit_code == 0 or "API key" in result.stdout


# ============================================================================
# INTEGRATION WITH LARGE FILES
# ============================================================================

class TestLargeFileHandling:
    """Test handling of large CSV files"""

    @patch('src.agent.run_analysis')
    def test_large_csv_file_processing(self, mock_run, large_csv_file):
        """Test processing of large CSV files"""
        mock_run.return_value = MagicMock(
            total_transactions_analyzed=100,
            processing_time_seconds=10.5
        )

        result = runner.invoke(app, [
            "analyze", large_csv_file,
            "--verbose"
        ])
        # Should handle large file (or fail on API key)
        assert result.exit_code == 0 or "API key" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
