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

    def test_default_llm_provider(self, sample_csv_file):
        """Test that default LLM provider is used when not specified"""
        result = runner.invoke(app, ["analyze", sample_csv_file])
        # Should use default from environment (anthropic)
        assert result.exit_code == 0

    def test_anthropic_provider_selection(self, sample_csv_file):
        """Test selecting Anthropic as LLM provider with real API"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "anthropic",
            "--model", "claude-3-5-sonnet-20241022"
        ])
        # Should successfully analyze with real LLM
        assert result.exit_code == 0

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

    def test_valid_anthropic_reasoning_models(self, sample_csv_file):
        """Test all valid Anthropic reasoning models with real API"""
        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929"
        ]
        # Test with one model (testing all would be too expensive)
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--provider", "anthropic",
            "--model", models[0]
        ])
        assert result.exit_code == 0, f"Model {models[0]} should work with real API"


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

    def test_output_path_option(self, sample_csv_file):
        """Test specifying custom output path with real LLM"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Should successfully create output file
            assert result.exit_code == 0
            assert os.path.exists(output_path)

    def test_threshold_option(self, sample_csv_file):
        """Test specifying custom threshold amount with real LLM"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--threshold", "500.0"
        ])
        assert result.exit_code == 0

    def test_negative_threshold_rejected(self, sample_csv_file):
        """Test that negative threshold is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--threshold", "-100.0"
        ])
        # Should fail with validation error
        assert result.exit_code != 0

    def test_currency_option(self, sample_csv_file):
        """Test specifying different base currency with real LLM"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--currency", "USD"
        ])
        assert result.exit_code == 0

    def test_invalid_currency_rejected(self, sample_csv_file):
        """Test that invalid currency is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--currency", "XYZ"
        ])
        assert result.exit_code != 0

    def test_mcts_iterations_option(self, sample_csv_file):
        """Test specifying custom MCTS iterations with real LLM"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--mcts-iterations", "50"
        ])
        assert result.exit_code == 0

    def test_zero_mcts_iterations_rejected(self, sample_csv_file):
        """Test that zero MCTS iterations is rejected"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--mcts-iterations", "0"
        ])
        assert result.exit_code != 0

    def test_verbose_flag(self, sample_csv_file):
        """Test verbose output flag with real LLM"""
        result = runner.invoke(app, [
            "analyze", sample_csv_file,
            "--verbose"
        ])
        assert result.exit_code == 0


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


# ============================================================================
# OUTPUT GENERATION TESTS
# ============================================================================

class TestOutputGeneration:
    """Test output file generation"""

    def test_output_file_created(self, sample_csv_file):
        """Test that output file is created successfully with real LLM"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Output file should be created
            assert result.exit_code == 0
            assert os.path.exists(output_path)
            # Verify output file has content
            df = pd.read_csv(output_path)
            assert len(df) > 0

    def test_output_directory_created(self, sample_csv_file):
        """Test that output directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subdir", "results.csv")
            result = runner.invoke(app, [
                "analyze", sample_csv_file,
                "--output", output_path
            ])
            # Should create parent directory and output file
            assert result.exit_code == 0
            assert os.path.exists(output_path)


# ============================================================================
# INTEGRATION WITH LARGE FILES
# ============================================================================

class TestLargeFileHandling:
    """Test handling of large CSV files"""

    def test_large_csv_file_processing(self, large_csv_file):
        """Test processing of large CSV files with real LLM"""
        result = runner.invoke(app, [
            "analyze", large_csv_file,
            "--verbose",
            "--mcts-iterations", "10"  # Reduced iterations for faster testing
        ])
        # Should successfully handle large file
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
