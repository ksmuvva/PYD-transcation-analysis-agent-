"""
Synthetic Transaction Data Generator

Generates realistic financial transaction data for testing the MCTS engine and tools.
Includes various scenarios: normal transactions, suspicious patterns, fraud indicators, etc.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

from src.models import Currency


class SyntheticTransactionGenerator:
    """Generate synthetic transaction data for testing."""

    # Merchant categories with typical merchants
    MERCHANT_CATEGORIES = {
        "Business": [
            "Office Depot", "Staples", "Amazon Business", "LinkedIn Premium",
            "Salesforce", "Microsoft 365", "Adobe Creative Cloud", "Zoom Pro",
            "WeWork", "FedEx Office", "UPS Store", "Conference Registration"
        ],
        "Personal": [
            "Tesco", "Sainsbury's", "Waitrose", "Marks & Spencer",
            "Netflix", "Spotify", "Gym Membership", "Haircut",
            "Restaurant", "Coffee Shop", "Cinema", "Pharmacy"
        ],
        "Travel": [
            "British Airways", "Hilton Hotels", "Premier Inn", "Marriott",
            "Uber", "Trainline", "National Express", "Europcar",
            "Holiday Inn", "Airbnb", "Booking.com", "Expedia"
        ],
        "Entertainment": [
            "Vue Cinema", "Odeon", "Spotify Premium", "Apple Music",
            "PlayStation Store", "Steam", "Nintendo eShop", "Netflix",
            "Disney+", "Amazon Prime Video", "Concert Tickets", "Theatre"
        ],
        "Investment": [
            "Vanguard", "Fidelity", "Charles Schwab", "Trading 212",
            "Hargreaves Lansdown", "Interactive Brokers", "eToro",
            "Nutmeg", "Wealthsimple", "Freetrade"
        ],
        "Gambling": [
            "Bet365", "William Hill", "Ladbrokes", "Paddy Power",
            "Sky Bet", "888 Casino", "PokerStars", "Betfair"
        ],
        "Suspicious": [
            "Crypto Exchange XYZ", "Offshore Payment Processor",
            "Unknown Merchant", "Wire Transfer Service",
            "Gift Card Vendor", "Money Transfer Service",
            "Anonymous Payment Gateway", "Unregistered Business"
        ]
    }

    # Fraud risk indicators
    FRAUD_INDICATORS = {
        "CRITICAL": {
            "amount_ranges": [(10000, 100000)],
            "merchants": MERCHANT_CATEGORIES["Suspicious"],
            "patterns": ["rapid_succession", "unusual_location", "high_value"]
        },
        "HIGH": {
            "amount_ranges": [(5000, 15000)],
            "merchants": MERCHANT_CATEGORIES["Gambling"] + MERCHANT_CATEGORIES["Suspicious"][:3],
            "patterns": ["multiple_small", "unusual_time", "new_merchant"]
        },
        "MEDIUM": {
            "amount_ranges": [(1000, 5000)],
            "merchants": ["Online Retailer Unknown", "New Subscription Service"],
            "patterns": ["increased_spending", "unusual_category"]
        },
        "LOW": {
            "amount_ranges": [(0, 1000)],
            "merchants": [],  # Normal merchants
            "patterns": []
        }
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with optional seed for reproducibility."""
        random.seed(seed)
        self.transaction_counter = 1000

    def generate_transaction(
        self,
        category: str | None = None,
        fraud_risk: str | None = None,
        amount_range: tuple[float, float] | None = None,
        currency: Currency = Currency.GBP
    ) -> Dict[str, Any]:
        """
        Generate a single synthetic transaction.

        Args:
            category: Transaction category (if None, random)
            fraud_risk: Fraud risk level (if None, mostly LOW)
            amount_range: Min/max amount (if None, based on category)
            currency: Transaction currency

        Returns:
            Transaction dictionary
        """
        # Select category
        if category is None:
            # Weight toward normal categories
            weights = [0.3, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05]
            category = random.choices(
                list(self.MERCHANT_CATEGORIES.keys()),
                weights=weights
            )[0]

        # Select merchant
        merchants = self.MERCHANT_CATEGORIES[category]
        merchant = random.choice(merchants) if merchants else "Unknown Merchant"

        # Determine fraud risk if not specified
        if fraud_risk is None:
            if category == "Suspicious":
                fraud_risk = random.choice(["HIGH", "CRITICAL"])
            elif category == "Gambling":
                fraud_risk = random.choice(["MEDIUM", "HIGH"])
            else:
                # Mostly low risk for normal categories
                fraud_risk = random.choices(
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    weights=[0.85, 0.10, 0.04, 0.01]
                )[0]

        # Generate amount based on category or range
        if amount_range:
            min_amount, max_amount = amount_range
        elif fraud_risk in self.FRAUD_INDICATORS:
            ranges = self.FRAUD_INDICATORS[fraud_risk]["amount_ranges"]
            min_amount, max_amount = random.choice(ranges) if ranges else (10, 500)
        else:
            # Default range based on category
            category_ranges = {
                "Business": (50, 2000),
                "Personal": (5, 200),
                "Travel": (100, 3000),
                "Entertainment": (10, 150),
                "Investment": (500, 10000),
                "Gambling": (20, 5000),
                "Suspicious": (1000, 50000)
            }
            min_amount, max_amount = category_ranges.get(category, (10, 500))

        amount = round(random.uniform(min_amount, max_amount), 2)

        # Generate date (last 90 days)
        days_ago = random.randint(0, 90)
        date = datetime.now() - timedelta(days=days_ago)

        # Generate description
        descriptions = {
            "Business": ["Office supplies", "Software subscription", "Conference fee", "Business travel"],
            "Personal": ["Groceries", "Shopping", "Monthly subscription", "Dining"],
            "Travel": ["Hotel booking", "Flight ticket", "Car rental", "Train ticket"],
            "Entertainment": ["Movie tickets", "Streaming service", "Gaming", "Concert"],
            "Investment": ["Stock purchase", "Fund investment", "Trading fee", "Portfolio rebalance"],
            "Gambling": ["Sports bet", "Casino gaming", "Online poker", "Lottery"],
            "Suspicious": ["Large transfer", "Crypto purchase", "Unknown payment", "Wire transfer"]
        }
        description = random.choice(descriptions.get(category, ["Transaction"]))

        # Generate transaction ID
        transaction_id = f"TX{self.transaction_counter:06d}"
        self.transaction_counter += 1

        return {
            "transaction_id": transaction_id,
            "amount": amount,
            "currency": currency.value,
            "date": date.strftime("%Y-%m-%d"),
            "merchant": merchant,
            "category": category,
            "description": description,
            "expected_fraud_risk": fraud_risk,  # Ground truth for testing
            "expected_classification": category  # Ground truth for testing
        }

    def generate_batch(
        self,
        count: int,
        category_distribution: Dict[str, float] | None = None,
        fraud_distribution: Dict[str, float] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of transactions.

        Args:
            count: Number of transactions to generate
            category_distribution: Category weights (if None, use defaults)
            fraud_distribution: Fraud risk weights (if None, use defaults)

        Returns:
            List of transaction dictionaries
        """
        transactions = []

        # Default distributions
        if category_distribution is None:
            category_distribution = {
                "Business": 0.30,
                "Personal": 0.30,
                "Travel": 0.15,
                "Entertainment": 0.10,
                "Investment": 0.05,
                "Gambling": 0.05,
                "Suspicious": 0.05
            }

        if fraud_distribution is None:
            fraud_distribution = {
                "LOW": 0.80,
                "MEDIUM": 0.12,
                "HIGH": 0.06,
                "CRITICAL": 0.02
            }

        for _ in range(count):
            # Sample category and fraud risk
            category = random.choices(
                list(category_distribution.keys()),
                weights=list(category_distribution.values())
            )[0]

            fraud_risk = random.choices(
                list(fraud_distribution.keys()),
                weights=list(fraud_distribution.values())
            )[0]

            transaction = self.generate_transaction(
                category=category,
                fraud_risk=fraud_risk
            )
            transactions.append(transaction)

        return transactions

    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge case transactions for testing."""
        edge_cases = []

        # Very small amount
        edge_cases.append(self.generate_transaction(
            category="Personal",
            amount_range=(0.01, 0.10),
            fraud_risk="LOW"
        ))

        # Very large amount
        edge_cases.append(self.generate_transaction(
            category="Suspicious",
            amount_range=(50000, 100000),
            fraud_risk="CRITICAL"
        ))

        # Exactly at threshold (250 GBP)
        edge_cases.append(self.generate_transaction(
            category="Business",
            amount_range=(250.00, 250.00),
            fraud_risk="LOW"
        ))

        # Just below threshold
        edge_cases.append(self.generate_transaction(
            category="Personal",
            amount_range=(249.99, 249.99),
            fraud_risk="LOW"
        ))

        # Just above threshold
        edge_cases.append(self.generate_transaction(
            category="Business",
            amount_range=(250.01, 250.01),
            fraud_risk="LOW"
        ))

        # Multiple currencies
        for currency in [Currency.USD, Currency.EUR, Currency.JPY]:
            edge_cases.append(self.generate_transaction(
                category="Travel",
                currency=currency,
                fraud_risk="LOW"
            ))

        # High-risk but small amount
        edge_cases.append(self.generate_transaction(
            category="Gambling",
            amount_range=(10, 50),
            fraud_risk="MEDIUM"
        ))

        # Low-risk but large amount
        edge_cases.append(self.generate_transaction(
            category="Investment",
            amount_range=(10000, 20000),
            fraud_risk="LOW"
        ))

        return edge_cases

    def generate_dataframe(
        self,
        count: int = 100,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of transactions.

        Args:
            count: Number of regular transactions
            include_edge_cases: Whether to include edge cases

        Returns:
            pandas DataFrame
        """
        transactions = self.generate_batch(count)

        if include_edge_cases:
            transactions.extend(self.generate_edge_cases())

        # Remove ground truth columns (used internally for validation)
        df_data = []
        for txn in transactions:
            row = {k: v for k, v in txn.items()
                   if k not in ["expected_fraud_risk", "expected_classification"]}
            df_data.append(row)

        return pd.DataFrame(df_data)

    def generate_fraud_scenario(
        self,
        scenario_type: str
    ) -> List[Dict[str, Any]]:
        """
        Generate specific fraud scenario patterns.

        Args:
            scenario_type: One of "rapid_succession", "unusual_amounts",
                          "suspicious_merchants", "mixed"

        Returns:
            List of related transactions forming a pattern
        """
        transactions = []

        if scenario_type == "rapid_succession":
            # Multiple transactions in short time
            base_date = datetime.now()
            for i in range(5):
                txn = self.generate_transaction(
                    category="Suspicious",
                    fraud_risk="HIGH"
                )
                # Override date to be within same hour
                txn["date"] = (base_date + timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S")
                transactions.append(txn)

        elif scenario_type == "unusual_amounts":
            # Gradually increasing amounts (testing)
            for amount in [100, 500, 1000, 5000, 10000]:
                txn = self.generate_transaction(
                    category="Suspicious",
                    amount_range=(amount, amount),
                    fraud_risk="HIGH" if amount > 1000 else "MEDIUM"
                )
                transactions.append(txn)

        elif scenario_type == "suspicious_merchants":
            # All suspicious category
            for _ in range(10):
                txn = self.generate_transaction(
                    category="Suspicious",
                    fraud_risk=random.choice(["HIGH", "CRITICAL"])
                )
                transactions.append(txn)

        elif scenario_type == "mixed":
            # Mix of normal and fraudulent
            for _ in range(5):
                transactions.append(self.generate_transaction(
                    category="Personal",
                    fraud_risk="LOW"
                ))
            for _ in range(3):
                transactions.append(self.generate_transaction(
                    category="Suspicious",
                    fraud_risk="CRITICAL"
                ))

        return transactions


# Convenience function
def create_test_dataset(
    size: str = "small",
    include_fraud: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a test dataset of specified size.

    Args:
        size: "tiny" (10), "small" (100), "medium" (500), "large" (1000)
        include_fraud: Include fraudulent transactions
        seed: Random seed for reproducibility

    Returns:
        pandas DataFrame
    """
    sizes = {
        "tiny": 10,
        "small": 100,
        "medium": 500,
        "large": 1000
    }

    count = sizes.get(size, 100)
    generator = SyntheticTransactionGenerator(seed=seed)

    if include_fraud:
        return generator.generate_dataframe(count=count, include_edge_cases=True)
    else:
        # Only low-risk transactions
        transactions = []
        for _ in range(count):
            txn = generator.generate_transaction(
                category=random.choice(["Business", "Personal", "Travel"]),
                fraud_risk="LOW"
            )
            transactions.append({k: v for k, v in txn.items()
                               if k not in ["expected_fraud_risk", "expected_classification"]})
        return pd.DataFrame(transactions)


if __name__ == "__main__":
    # Generate sample datasets for manual inspection
    generator = SyntheticTransactionGenerator()

    # Small dataset
    df_small = create_test_dataset("small")
    print(f"\nSmall dataset: {len(df_small)} transactions")
    print(df_small.head(10))

    # Edge cases
    edge_cases = generator.generate_edge_cases()
    print(f"\nEdge cases: {len(edge_cases)} transactions")
    for ec in edge_cases[:3]:
        print(f"  {ec['transaction_id']}: {ec['amount']} {ec['currency']} - {ec['merchant']}")

    # Fraud scenario
    fraud_scenario = generator.generate_fraud_scenario("rapid_succession")
    print(f"\nFraud scenario (rapid succession): {len(fraud_scenario)} transactions")
    for fs in fraud_scenario:
        print(f"  {fs['transaction_id']}: {fs['amount']} {fs['currency']} - {fs['merchant']}")
