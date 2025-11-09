#!/usr/bin/env python3
"""
OpenAI MCTS Integration Test Runner

This script runs comprehensive integration tests for the MCTS engine using real OpenAI API.
Includes options for different test modes and models.

Usage:
    # Run all tests (warning: expensive!)
    python run_openai_tests.py --all

    # Run fast tests only
    python run_openai_tests.py --fast

    # Run specific model tests
    python run_openai_tests.py --model o1-mini

    # Run specific test category
    python run_openai_tests.py --category mcts

    # Dry run (validate setup only)
    python run_openai_tests.py --dry-run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-key-here")
        return False

    if len(api_key) < 20:
        print(f"⚠️  WARNING: API key looks invalid (too short): {api_key[:10]}...")
        return False

    print(f"✅ OpenAI API key found: {api_key[:15]}...")
    return True


def estimate_cost(test_mode: str, model: str) -> dict:
    """Estimate approximate API cost for test run."""
    # Rough estimates based on typical usage
    costs = {
        "o1-mini": {
            "fast": {"calls": 20, "tokens": 20000, "cost_usd": 0.30},
            "all": {"calls": 100, "tokens": 100000, "cost_usd": 1.50},
        },
        "o1-preview": {
            "fast": {"calls": 20, "tokens": 20000, "cost_usd": 2.00},
            "all": {"calls": 100, "tokens": 100000, "cost_usd": 10.00},
        },
        "o1": {
            "fast": {"calls": 20, "tokens": 20000, "cost_usd": 5.00},
            "all": {"calls": 100, "tokens": 100000, "cost_usd": 25.00},
        },
    }

    model_key = model if model in costs else "o1-mini"
    mode_key = test_mode if test_mode in ["fast", "all"] else "fast"

    return costs[model_key][mode_key]


def run_pytest(args_list: list) -> int:
    """Run pytest with specified arguments."""
    cmd = ["pytest"] + args_list
    print(f"\n🔧 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenAI MCTS integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (fast, o1-mini only)
  python run_openai_tests.py --fast --model o1-mini

  # Comprehensive test (all tests, all models)
  python run_openai_tests.py --all

  # Test specific category
  python run_openai_tests.py --category TestMCTSEngineWithOpenAI

  # Dry run (check setup without running tests)
  python run_openai_tests.py --dry-run
        """
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run fast tests only (reduced iterations, quick models)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests including slow and expensive ones'
    )

    parser.add_argument(
        '--model',
        choices=['o1-mini', 'o1-preview', 'o1', 'o3-mini'],
        default='o1-mini',
        help='OpenAI model to test (default: o1-mini)'
    )

    parser.add_argument(
        '--category',
        help='Run specific test category (e.g., TestFilterToolWithRealAPI)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate setup without running tests'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--markers',
        help='Additional pytest markers (e.g., "not slow")'
    )

    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("OpenAI MCTS Integration Test Runner")
    print("=" * 70)

    # Check API key
    if not check_api_key():
        return 1

    # Dry run - just validate setup
    if args.dry_run:
        print("\n✅ DRY RUN: Setup validated successfully")
        print("\nTo run actual tests, use:")
        print("  python run_openai_tests.py --fast")
        return 0

    # Determine test mode
    if args.all:
        test_mode = "all"
    elif args.fast:
        test_mode = "fast"
    else:
        # Default to fast
        test_mode = "fast"

    # Estimate cost
    estimate = estimate_cost(test_mode, args.model)
    print(f"\n📊 Test Configuration:")
    print(f"  Mode: {test_mode}")
    print(f"  Model: {args.model}")
    print(f"  Category: {args.category or 'all'}")
    print(f"\n💰 Estimated Cost:")
    print(f"  API Calls: ~{estimate['calls']}")
    print(f"  Tokens: ~{estimate['tokens']:,}")
    print(f"  Cost: ~${estimate['cost_usd']:.2f} USD")

    # Confirm
    if not args.fast:
        response = input("\n⚠️  Proceed with tests? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Tests cancelled")
            return 0

    # Build pytest arguments
    pytest_args = [
        "tests/test_openai_mcts_integration.py",
        "-v" if args.verbose else "-q",
        "--tb=short",
        "-s",  # Show print output
        f"--model={args.model}",
    ]

    # Add markers
    if args.fast:
        # Run only fast tests
        pytest_args.extend(["-m", "not slow"])

    if args.markers:
        pytest_args.extend(["-m", args.markers])

    # Add category filter
    if args.category:
        pytest_args.extend(["-k", args.category])

    # Run tests
    print("\n" + "=" * 70)
    print("Running Tests...")
    print("=" * 70)

    returncode = run_pytest(pytest_args)

    # Summary
    print("\n" + "=" * 70)
    if returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {returncode}")
    print("=" * 70)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
