"""
Generate 25 synthetic transaction datasets with ground truth labels for testing.
Each dataset has different characteristics to test various aspects of the agent.
"""

import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

# Set seed for reproducibility
random.seed(42)

# Dataset categories and fraud indicators
CATEGORIES = {
    "Business Expense": ["office supplies", "software subscription", "professional services", "consulting", "marketing"],
    "Personal": ["grocery", "restaurant", "entertainment", "clothing", "pharmacy"],
    "Travel": ["airline", "hotel", "car rental", "travel agency", "taxi"],
    "Subscription": ["streaming service", "gym membership", "magazine", "online service"],
    "Utilities": ["electricity", "water", "internet", "phone bill"],
    "Healthcare": ["hospital", "clinic", "medical equipment", "pharmacy"],
    "Education": ["university", "online course", "books", "tuition"],
    "Investment": ["stock purchase", "mutual fund", "crypto exchange"],
}

FRAUD_INDICATORS = {
    "LOW": {
        "amount_range": (10, 200),
        "merchants": ["Amazon", "Walmart", "Target", "Local Store", "Pharmacy"],
        "descriptions": ["routine purchase", "regular shopping", "monthly subscription"],
    },
    "MEDIUM": {
        "amount_range": (200, 500),
        "merchants": ["Unknown Vendor", "International Store", "New Merchant"],
        "descriptions": ["unusual location", "first time purchase", "late night transaction"],
    },
    "HIGH": {
        "amount_range": (500, 2000),
        "merchants": ["Suspicious Store XYZ", "Offshore Vendor", "Unknown Inc"],
        "descriptions": ["unusual amount", "rapid succession", "duplicate charge"],
    },
    "CRITICAL": {
        "amount_range": (2000, 10000),
        "merchants": ["Fake Merchant 123", "Scam Corp", "Unknown Overseas"],
        "descriptions": ["stolen card pattern", "multiple failed attempts", "unusual country"],
    },
}

CURRENCIES = ["GBP", "USD", "EUR", "JPY", "CAD", "AUD"]

# Exchange rates to GBP
EXCHANGE_RATES = {
    "GBP": 1.0,
    "USD": 0.79,
    "EUR": 0.85,
    "JPY": 0.0052,
    "CAD": 0.58,
    "AUD": 0.52,
}


def generate_transaction(tx_id, category, fraud_level, date_offset=0, currency="GBP"):
    """Generate a single transaction with ground truth labels."""

    # Generate amount based on fraud level
    amount_min, amount_max = FRAUD_INDICATORS[fraud_level]["amount_range"]
    amount = round(random.uniform(amount_min, amount_max), 2)

    # Select merchant and description
    merchant = random.choice(FRAUD_INDICATORS[fraud_level]["merchants"])
    description_suffix = random.choice(FRAUD_INDICATORS[fraud_level]["descriptions"])
    category_desc = random.choice(CATEGORIES[category])
    description = f"{category_desc} - {description_suffix}"

    # Generate date
    base_date = datetime.now() - timedelta(days=date_offset)
    date_str = base_date.strftime("%Y-%m-%d")

    # Calculate GBP equivalent
    gbp_equivalent = round(amount * EXCHANGE_RATES[currency], 2)

    return {
        "transaction_id": f"TXN{tx_id:06d}",
        "amount": amount,
        "currency": currency,
        "date": date_str,
        "merchant": merchant,
        "description": description,
        "ground_truth_category": category,
        "ground_truth_fraud_level": fraud_level,
        "ground_truth_gbp_amount": gbp_equivalent,
        "ground_truth_above_threshold": gbp_equivalent >= 250.0,
    }


def save_dataset(transactions, dataset_num, dataset_type, output_dir):
    """Save dataset and ground truth to files."""

    # Create DataFrame
    df = pd.DataFrame(transactions)

    # Save CSV (without ground truth columns for testing)
    csv_df = df[["transaction_id", "amount", "currency", "date", "merchant", "description"]]
    csv_path = output_dir / f"dataset_{dataset_num:02d}_{dataset_type}.csv"
    csv_df.to_csv(csv_path, index=False)

    # Save ground truth
    ground_truth = df[["transaction_id", "ground_truth_category", "ground_truth_fraud_level",
                       "ground_truth_gbp_amount", "ground_truth_above_threshold"]].to_dict(orient="records")
    gt_path = output_dir / f"dataset_{dataset_num:02d}_{dataset_type}_ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {dataset_type}: {csv_path.name} ({len(transactions)} transactions)")
    return csv_path, gt_path


def main():
    output_dir = Path(__file__).parent / "synthetic_datasets"
    output_dir.mkdir(exist_ok=True)

    dataset_num = 1

    # ========== DATASETS 1-5: SMALL BALANCED (5 transactions each) ==========
    print("\n=== Generating Small Balanced Datasets (1-5) ===")
    for i in range(5):
        transactions = []
        categories = list(CATEGORIES.keys())[:5]
        fraud_levels = ["LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

        for j in range(5):
            tx = generate_transaction(
                tx_id=dataset_num * 1000 + j,
                category=categories[j],
                fraud_level=fraud_levels[j],
                date_offset=random.randint(1, 90),
                currency=random.choice(CURRENCIES),
            )
            transactions.append(tx)

        save_dataset(transactions, dataset_num, "small_balanced", output_dir)
        dataset_num += 1

    # ========== DATASETS 6-10: MEDIUM VARIED (20 transactions each) ==========
    print("\n=== Generating Medium Varied Datasets (6-10) ===")
    for i in range(5):
        transactions = []

        for j in range(20):
            category = random.choice(list(CATEGORIES.keys()))
            fraud_level = random.choices(["LOW", "MEDIUM", "HIGH", "CRITICAL"], weights=[0.5, 0.3, 0.15, 0.05])[0]

            tx = generate_transaction(
                tx_id=dataset_num * 1000 + j,
                category=category,
                fraud_level=fraud_level,
                date_offset=random.randint(1, 180),
                currency=random.choice(CURRENCIES),
            )
            transactions.append(tx)

        save_dataset(transactions, dataset_num, "medium_varied", output_dir)
        dataset_num += 1

    # ========== DATASETS 11-15: LARGE REALISTIC (50 transactions each) ==========
    print("\n=== Generating Large Realistic Datasets (11-15) ===")
    for i in range(5):
        transactions = []

        for j in range(50):
            category = random.choice(list(CATEGORIES.keys()))
            fraud_level = random.choices(["LOW", "MEDIUM", "HIGH", "CRITICAL"], weights=[0.7, 0.2, 0.08, 0.02])[0]

            tx = generate_transaction(
                tx_id=dataset_num * 1000 + j,
                category=category,
                fraud_level=fraud_level,
                date_offset=random.randint(1, 365),
                currency=random.choice(CURRENCIES),
            )
            transactions.append(tx)

        save_dataset(transactions, dataset_num, "large_realistic", output_dir)
        dataset_num += 1

    # ========== DATASET 16: ALL LOW FRAUD (10 transactions) ==========
    print("\n=== Generating Edge Case Datasets (16-25) ===")
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category="Personal",
            fraud_level="LOW",
            date_offset=random.randint(1, 30),
            currency="GBP",
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "all_low_fraud", output_dir)
    dataset_num += 1

    # ========== DATASET 17: ALL CRITICAL FRAUD (10 transactions) ==========
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level="CRITICAL",
            date_offset=random.randint(1, 30),
            currency=random.choice(CURRENCIES),
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "all_critical_fraud", output_dir)
    dataset_num += 1

    # ========== DATASET 18: SINGLE CURRENCY - USD (15 transactions) ==========
    transactions = []
    for j in range(15):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level=random.choice(["LOW", "MEDIUM", "HIGH"]),
            date_offset=random.randint(1, 60),
            currency="USD",
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "single_currency_usd", output_dir)
    dataset_num += 1

    # ========== DATASET 19: MULTI-CURRENCY MIX (20 transactions) ==========
    transactions = []
    for j in range(20):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            date_offset=random.randint(1, 90),
            currency=CURRENCIES[j % len(CURRENCIES)],  # Cycle through all currencies
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "multi_currency_mix", output_dir)
    dataset_num += 1

    # ========== DATASET 20: BELOW THRESHOLD (10 transactions) ==========
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level="LOW",
            date_offset=random.randint(1, 30),
            currency="GBP",
        )
        # Force below threshold
        tx["amount"] = round(random.uniform(10, 100), 2)
        tx["ground_truth_gbp_amount"] = tx["amount"]
        tx["ground_truth_above_threshold"] = False
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "below_threshold", output_dir)
    dataset_num += 1

    # ========== DATASET 21: ABOVE THRESHOLD (10 transactions) ==========
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level=random.choice(["MEDIUM", "HIGH", "CRITICAL"]),
            date_offset=random.randint(1, 30),
            currency="GBP",
        )
        # Force above threshold
        tx["amount"] = round(random.uniform(300, 1000), 2)
        tx["ground_truth_gbp_amount"] = tx["amount"]
        tx["ground_truth_above_threshold"] = True
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "above_threshold", output_dir)
    dataset_num += 1

    # ========== DATASET 22: BUSINESS EXPENSES ONLY (15 transactions) ==========
    transactions = []
    for j in range(15):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category="Business Expense",
            fraud_level=random.choice(["LOW", "MEDIUM"]),
            date_offset=random.randint(1, 60),
            currency=random.choice(CURRENCIES),
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "business_only", output_dir)
    dataset_num += 1

    # ========== DATASET 23: TRAVEL EXPENSES ONLY (15 transactions) ==========
    transactions = []
    for j in range(15):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category="Travel",
            fraud_level=random.choice(["LOW", "MEDIUM", "HIGH"]),
            date_offset=random.randint(1, 60),
            currency=random.choice(["USD", "EUR", "GBP"]),
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "travel_only", output_dir)
    dataset_num += 1

    # ========== DATASET 24: RECENT TRANSACTIONS (10 transactions, last 7 days) ==========
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level=random.choice(["LOW", "MEDIUM", "HIGH"]),
            date_offset=random.randint(0, 7),  # Last 7 days
            currency=random.choice(CURRENCIES),
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "recent_transactions", output_dir)
    dataset_num += 1

    # ========== DATASET 25: OLD TRANSACTIONS (10 transactions, 1+ years ago) ==========
    transactions = []
    for j in range(10):
        tx = generate_transaction(
            tx_id=dataset_num * 1000 + j,
            category=random.choice(list(CATEGORIES.keys())),
            fraud_level=random.choice(["LOW", "MEDIUM"]),
            date_offset=random.randint(365, 730),  # 1-2 years ago
            currency=random.choice(CURRENCIES),
        )
        transactions.append(tx)
    save_dataset(transactions, dataset_num, "old_transactions", output_dir)
    dataset_num += 1

    print(f"\n✓ Successfully generated 25 synthetic datasets in {output_dir}")
    print("✓ Each dataset includes CSV file and ground truth JSON")


if __name__ == "__main__":
    main()
