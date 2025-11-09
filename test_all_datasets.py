"""
Comprehensive Dataset Testing Script
Tests the agent with all 25 synthetic datasets and calculates metrics
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import run_analysis
from src.config import AgentConfig, ConfigManager
from src.csv_processor import CSVProcessor


def load_ground_truth(gt_path):
    """Load ground truth from JSON file"""
    with open(gt_path, 'r') as f:
        return {item['transaction_id']: item for item in json.load(f)}


def calculate_metrics(results_df, ground_truth):
    """Calculate accuracy metrics for classification and fraud detection"""
    metrics = {
        'total_transactions': len(results_df),
        'transactions_analyzed': 0,
        'classification_metrics': {},
        'fraud_metrics': {},
        'errors': []
    }

    # Only analyze transactions that were processed (above threshold)
    analyzed = results_df[results_df['above_250_gbp'] == True]
    metrics['transactions_analyzed'] = len(analyzed)

    if len(analyzed) == 0:
        return metrics

    # Classification metrics
    correct_classifications = 0
    total_with_gt = 0

    # Fraud detection metrics - map to simple levels for comparison
    fraud_level_map = {
        'LOW': 0,
        'MEDIUM': 1,
        'HIGH': 2,
        'CRITICAL': 3
    }

    correct_fraud = 0
    fraud_total = 0
    fraud_errors = []

    for _, row in analyzed.iterrows():
        tx_id = row['transaction_id']
        if tx_id in ground_truth:
            gt = ground_truth[tx_id]
            total_with_gt += 1

            # Check classification
            pred_class = row.get('classification', '')
            gt_class = gt['ground_truth_category']

            # Fuzzy matching for classification (Business Expense matches Business, etc.)
            if pred_class and gt_class:
                if pred_class.lower() in gt_class.lower() or gt_class.lower() in pred_class.lower():
                    correct_classifications += 1

            # Check fraud detection
            pred_fraud = row.get('fraud_risk', '')
            gt_fraud = gt['ground_truth_fraud_level']

            if pred_fraud and gt_fraud:
                fraud_total += 1
                # Exact match
                if pred_fraud == gt_fraud:
                    correct_fraud += 1
                # Allow ±1 level tolerance
                elif abs(fraud_level_map.get(pred_fraud, 0) - fraud_level_map.get(gt_fraud, 0)) <= 1:
                    correct_fraud += 0.5  # Partial credit
                else:
                    fraud_errors.append({
                        'tx_id': tx_id,
                        'predicted': pred_fraud,
                        'actual': gt_fraud
                    })

    if total_with_gt > 0:
        metrics['classification_metrics'] = {
            'total': total_with_gt,
            'correct': correct_classifications,
            'accuracy': correct_classifications / total_with_gt
        }

    if fraud_total > 0:
        metrics['fraud_metrics'] = {
            'total': fraud_total,
            'correct': correct_fraud,
            'accuracy': correct_fraud / fraud_total,
            'major_errors': len(fraud_errors),
            'errors': fraud_errors[:5]  # Show first 5 errors
        }

    return metrics


def test_dataset(dataset_path, ground_truth_path, output_dir):
    """Test a single dataset"""
    print(f"\nTesting: {dataset_path.name}")
    print("-" * 80)

    try:
        # Load ground truth
        ground_truth = load_ground_truth(ground_truth_path)
        print(f"✓ Loaded ground truth: {len(ground_truth)} transactions")

        # Run analysis
        start_time = time.time()
        config = ConfigManager.load_from_env(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            mcts_iterations=10  # Reduced for faster testing
        )

        output_path = output_dir / f"analyzed_{dataset_path.name}"
        report = run_analysis(str(dataset_path), str(output_path), config)

        elapsed = time.time() - start_time

        # Load results
        results_df = pd.read_csv(output_path)

        # Calculate metrics
        metrics = calculate_metrics(results_df, ground_truth)

        result = {
            'dataset': dataset_path.name,
            'status': 'success',
            'elapsed_time': elapsed,
            'report': {
                'total_transactions': report.total_transactions,
                'high_value_count': report.high_value_count,
                'high_risk_count': report.high_risk_count,
            },
            'metrics': metrics
        }

        # Print summary
        print(f"✓ Analysis complete in {elapsed:.2f}s")
        print(f"  Transactions analyzed: {metrics['transactions_analyzed']}/{metrics['total_transactions']}")

        if metrics['classification_metrics']:
            cm = metrics['classification_metrics']
            print(f"  Classification accuracy: {cm['accuracy']:.1%} ({cm['correct']}/{cm['total']})")

        if metrics['fraud_metrics']:
            fm = metrics['fraud_metrics']
            print(f"  Fraud detection accuracy: {fm['accuracy']:.1%} ({fm['correct']:.1f}/{fm['total']})")

        return result

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_path.name,
            'status': 'error',
            'error': str(e)
        }


def main():
    print("=" * 80)
    print("COMPREHENSIVE DATASET TESTING")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")

    # Setup
    datasets_dir = Path(__file__).parent / "tests" / "synthetic_datasets"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)

    # Find all datasets
    csv_files = sorted(datasets_dir.glob("dataset_*.csv"))
    print(f"\nFound {len(csv_files)} datasets to test")

    # Test each dataset
    results = []
    for csv_file in csv_files:
        # Find corresponding ground truth
        gt_file = csv_file.parent / csv_file.name.replace('.csv', '_ground_truth.json')

        if not gt_file.exists():
            print(f"⚠ Skipping {csv_file.name} - no ground truth found")
            continue

        result = test_dataset(csv_file, gt_file, output_dir)
        results.append(result)

        # Brief pause between tests to avoid rate limiting
        time.sleep(2)

    # Generate overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print(f"\nDatasets tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        # Overall metrics
        total_analyzed = sum(r['metrics']['transactions_analyzed'] for r in successful)
        total_transactions = sum(r['metrics']['total_transactions'] for r in successful)

        print(f"\nTotal transactions: {total_transactions}")
        print(f"Transactions analyzed: {total_analyzed}")

        # Classification metrics
        class_correct = sum(r['metrics']['classification_metrics'].get('correct', 0) for r in successful if r['metrics']['classification_metrics'])
        class_total = sum(r['metrics']['classification_metrics'].get('total', 0) for r in successful if r['metrics']['classification_metrics'])

        if class_total > 0:
            print(f"\nOverall Classification Accuracy: {class_correct/class_total:.1%} ({class_correct:.0f}/{class_total})")

        # Fraud metrics
        fraud_correct = sum(r['metrics']['fraud_metrics'].get('correct', 0) for r in successful if r['metrics']['fraud_metrics'])
        fraud_total = sum(r['metrics']['fraud_metrics'].get('total', 0) for r in successful if r['metrics']['fraud_metrics'])

        if fraud_total > 0:
            print(f"Overall Fraud Detection Accuracy: {fraud_correct/fraud_total:.1%} ({fraud_correct:.1f}/{fraud_total})")

    # Save detailed results
    results_file = output_dir / "comprehensive_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_datasets': len(results),
                'successful': len(successful),
                'failed': len(failed)
            },
            'results': results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
