"""
Dataset generator for evaluation test cases (REQ-EVAL-004).

Generates minimum 100 cases per tool with 20% adversarial examples.
"""

import random
from datetime import datetime, timedelta
from typing import List

from evals.models import Case, ExpectedOutput, CaseMetadata


class DatasetGenerator:
    """
    Generates synthetic test cases for evaluation (REQ-EVAL-004).

    Implements:
    - Minimum 100 cases per tool
    - 20% adversarial examples (edge amounts, ambiguous MCCs, synthetic fraud)
    """

    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.random_seed = random_seed
        random.seed(random_seed)

    def generate_filter_cases(self, count: int = 100) -> List[Case]:
        """
        Generate test cases for Tool 1: Filter transactions above 250 GBP.

        Args:
            count: Number of cases to generate (default 100)

        Returns:
            List of Case objects
        """
        cases = []
        adversarial_count = int(count * 0.2)  # 20% adversarial

        # Regular cases (80%)
        for i in range(count - adversarial_count):
            amount, currency, expected_gbp, is_above = self._generate_regular_filter_amount(i)

            cases.append(Case(
                inputs={
                    "tx_id": f"FILTER_REG_{i:03d}",
                    "amount": amount,
                    "currency": currency,
                    "merchant": f"Regular Merchant {i}",
                    "description": f"Regular transaction {i}",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=is_above,
                    expected_gbp_amount=expected_gbp,
                    tool_2_classification="Personal",
                    tool_3_fraud_risk="LOW",
                    tool_3_confidence=0.85,
                    tool_4_columns_complete=True,
                    expected_min_iterations=100,
                    expected_min_reward=1.0,
                    expected_min_path_length=1,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="regular",
                    labeled_by="synthetic_generator",
                    is_fraud=False,
                    is_adversarial=False,
                )
            ))

        # Adversarial cases (20%) - edge amounts near 250 threshold
        for i in range(adversarial_count):
            amount, currency, expected_gbp, is_above = self._generate_adversarial_filter_amount(i)

            cases.append(Case(
                inputs={
                    "tx_id": f"FILTER_ADV_{i:03d}",
                    "amount": amount,
                    "currency": currency,
                    "merchant": f"Edge Case Merchant {i}",
                    "description": f"Adversarial threshold test {i}",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=is_above,
                    expected_gbp_amount=expected_gbp,
                    tool_2_classification="Personal",
                    tool_3_fraud_risk="LOW",
                    tool_3_confidence=0.85,
                    tool_4_columns_complete=True,
                    expected_min_iterations=100,
                    expected_min_reward=1.0,
                    expected_min_path_length=1,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="adversarial_threshold",
                    labeled_by="synthetic_generator",
                    is_fraud=False,
                    is_adversarial=True,
                )
            ))

        return cases

    def generate_classification_cases(self, count: int = 100) -> List[Case]:
        """
        Generate test cases for Tool 2: Classify transactions.

        Args:
            count: Number of cases to generate (default 100)

        Returns:
            List of Case objects
        """
        cases = []
        adversarial_count = int(count * 0.2)
        categories = ["Business", "Personal", "Investment", "Gambling"]

        # Regular cases (80%)
        regular_per_category = (count - adversarial_count) // len(categories)
        for cat_idx, category in enumerate(categories):
            for i in range(regular_per_category):
                merchant, description = self._generate_category_merchant(category, i)

                cases.append(Case(
                    inputs={
                        "tx_id": f"CLASS_{category[:3].upper()}_{i:03d}",
                        "amount": random.uniform(50, 5000),
                        "currency": random.choice(["GBP", "USD", "EUR"]),
                        "merchant": merchant,
                        "description": description,
                        "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    },
                    expected_output=ExpectedOutput(
                        tool_1_filtered=random.choice([True, False]),
                        tool_2_classification=category,
                        tool_3_fraud_risk="LOW",
                        tool_3_confidence=0.85,
                        tool_4_columns_complete=True,
                        expected_min_iterations=500,
                        expected_min_reward=0.80,
                        expected_min_path_length=10,
                        max_acceptable_variance=0.05,
                    ),
                    metadata=CaseMetadata(
                        category=f"classification_{category.lower()}",
                        labeled_by="synthetic_generator",
                        is_fraud=False,
                        is_adversarial=False,
                    )
                ))

        # Adversarial cases (20%) - ambiguous classifications
        for i in range(adversarial_count):
            category, merchant, description = self._generate_ambiguous_classification(i)

            cases.append(Case(
                inputs={
                    "tx_id": f"CLASS_AMB_{i:03d}",
                    "amount": random.uniform(50, 5000),
                    "currency": random.choice(["GBP", "USD", "EUR"]),
                    "merchant": merchant,
                    "description": description,
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=random.choice([True, False]),
                    tool_2_classification=category,
                    tool_3_fraud_risk="LOW",
                    tool_3_confidence=0.60,  # Lower confidence for ambiguous
                    tool_4_columns_complete=True,
                    expected_min_iterations=500,
                    expected_min_reward=0.80,
                    expected_min_path_length=10,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="adversarial_ambiguous",
                    labeled_by="synthetic_generator",
                    is_fraud=False,
                    is_adversarial=True,
                )
            ))

        return cases

    def generate_fraud_cases(self, count: int = 100) -> List[Case]:
        """
        Generate test cases for Tool 3: Detect fraudulent transactions.

        Args:
            count: Number of cases to generate (default 100)

        Returns:
            List of Case objects
        """
        cases = []
        adversarial_count = int(count * 0.2)
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # Regular cases (80%)
        regular_per_level = (count - adversarial_count) // len(risk_levels)
        for level_idx, risk_level in enumerate(risk_levels):
            for i in range(regular_per_level):
                amount, merchant, indicators = self._generate_fraud_pattern(risk_level, i)
                is_fraud = (risk_level in ["HIGH", "CRITICAL"])

                cases.append(Case(
                    inputs={
                        "tx_id": f"FRAUD_{risk_level[:3]}_{i:03d}",
                        "amount": amount,
                        "currency": "GBP",
                        "merchant": merchant,
                        "description": f"Transaction with {risk_level} risk",
                        "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    },
                    expected_output=ExpectedOutput(
                        tool_1_filtered=amount >= 250,
                        tool_2_classification="Personal",
                        tool_3_fraud_risk=risk_level,
                        tool_3_confidence=0.90 if risk_level == "CRITICAL" else 0.75,
                        expected_fraud_indicators=indicators,
                        tool_4_columns_complete=True,
                        expected_min_iterations=1000,
                        expected_min_reward=0.90,
                        expected_min_path_length=10,
                        max_acceptable_variance=0.05,
                    ),
                    metadata=CaseMetadata(
                        category=f"fraud_{risk_level.lower()}",
                        labeled_by="compliance_team",
                        is_fraud=is_fraud,
                        is_adversarial=False,
                        mcc_code=self._get_mcc_for_risk(risk_level),
                    )
                ))

        # Adversarial cases (20%) - synthetic fraud patterns
        for i in range(adversarial_count):
            risk_level, amount, merchant, indicators = self._generate_synthetic_fraud(i)
            is_fraud = (risk_level in ["HIGH", "CRITICAL"])

            cases.append(Case(
                inputs={
                    "tx_id": f"FRAUD_SYN_{i:03d}",
                    "amount": amount,
                    "currency": random.choice(["GBP", "USD", "EUR", "JPY"]),
                    "merchant": merchant,
                    "description": f"Synthetic fraud pattern {i}",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=amount >= 250,
                    tool_2_classification="Personal",
                    tool_3_fraud_risk=risk_level,
                    tool_3_confidence=0.70,  # Lower confidence for synthetic
                    expected_fraud_indicators=indicators,
                    tool_4_columns_complete=True,
                    expected_min_iterations=1000,
                    expected_min_reward=0.90,
                    expected_min_path_length=10,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="adversarial_synthetic_fraud",
                    labeled_by="synthetic_generator",
                    is_fraud=is_fraud,
                    is_adversarial=True,
                    mcc_code=self._get_mcc_for_risk(risk_level),
                )
            ))

        return cases

    def generate_csv_cases(self, count: int = 100) -> List[Case]:
        """
        Generate test cases for Tool 4: Generate enhanced CSV.

        Args:
            count: Number of cases to generate (default 100)

        Returns:
            List of Case objects
        """
        cases = []
        adversarial_count = int(count * 0.2)

        # Regular cases (80%)
        for i in range(count - adversarial_count):
            cases.append(Case(
                inputs={
                    "tx_id": f"CSV_REG_{i:03d}",
                    "amount": random.uniform(50, 5000),
                    "currency": random.choice(["GBP", "USD", "EUR"]),
                    "merchant": f"Merchant {i}",
                    "description": f"CSV test transaction {i}",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=random.choice([True, False]),
                    tool_2_classification=random.choice(["Business", "Personal", "Investment", "Gambling"]),
                    tool_3_fraud_risk=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    tool_3_confidence=0.85,
                    tool_4_columns_complete=True,
                    expected_row_count=1,
                    expected_keywords_in_explanation=["classification", "fraud", "risk"],
                    expected_min_iterations=200,
                    expected_min_reward=1.0,
                    expected_min_path_length=1,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="csv_regular",
                    labeled_by="synthetic_generator",
                    is_fraud=False,
                    is_adversarial=False,
                )
            ))

        # Adversarial cases (20%) - edge cases for CSV generation
        for i in range(adversarial_count):
            cases.append(Case(
                inputs={
                    "tx_id": f"CSV_ADV_{i:03d}",
                    "amount": random.uniform(50, 5000),
                    "currency": random.choice(["GBP", "USD", "EUR"]),
                    "merchant": f"Special Chars £€$ {i}",
                    "description": f"Edge case with\nspecial\tcharacters",
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                expected_output=ExpectedOutput(
                    tool_1_filtered=random.choice([True, False]),
                    tool_2_classification=random.choice(["Business", "Personal", "Investment", "Gambling"]),
                    tool_3_fraud_risk=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    tool_3_confidence=0.85,
                    tool_4_columns_complete=True,
                    expected_row_count=1,
                    expected_keywords_in_explanation=["classification", "fraud"],
                    expected_min_iterations=200,
                    expected_min_reward=1.0,
                    expected_min_path_length=1,
                    max_acceptable_variance=0.05,
                ),
                metadata=CaseMetadata(
                    category="adversarial_special_chars",
                    labeled_by="synthetic_generator",
                    is_fraud=False,
                    is_adversarial=True,
                )
            ))

        return cases

    # Helper methods for data generation

    def _generate_regular_filter_amount(self, seed: int) -> tuple:
        """Generate regular (non-edge) filter test amounts."""
        random.seed(self.random_seed + seed)
        is_above = random.choice([True, False])

        if is_above:
            # Well above threshold
            amount = random.uniform(300, 10000)
            currency = "GBP"
            expected_gbp = amount
        else:
            # Well below threshold
            amount = random.uniform(10, 200)
            currency = "GBP"
            expected_gbp = amount

        return amount, currency, expected_gbp, is_above

    def _generate_adversarial_filter_amount(self, seed: int) -> tuple:
        """Generate adversarial (edge) filter test amounts near 250 threshold."""
        random.seed(self.random_seed + seed)

        # Amounts very close to 250 GBP threshold
        if random.choice([True, False]):
            amount = random.uniform(249.0, 249.99)  # Just below
            is_above = False
        else:
            amount = random.uniform(250.0, 251.0)  # Just above
            is_above = True

        currency = "GBP"
        expected_gbp = amount

        return amount, currency, expected_gbp, is_above

    def _generate_category_merchant(self, category: str, seed: int) -> tuple:
        """Generate merchant and description for a given category."""
        random.seed(self.random_seed + seed)

        patterns = {
            "Business": [
                ("ABC Corp Ltd", "Office supplies purchase"),
                ("XYZ Inc", "Business software subscription"),
                ("Professional Services Corp", "Consulting services"),
            ],
            "Personal": [
                ("Supermarket", "Grocery shopping"),
                ("Coffee Shop", "Coffee and snacks"),
                ("Local Restaurant", "Dinner"),
            ],
            "Investment": [
                ("Trading Platform", "Stock purchase"),
                ("Investment Fund", "Fund contribution"),
                ("Broker Services", "Investment management fee"),
            ],
            "Gambling": [
                ("Online Casino", "Casino deposit"),
                ("Betting Site", "Sports bet"),
                ("Poker Platform", "Tournament entry"),
            ],
        }

        merchant, description = random.choice(patterns[category])
        return merchant, description

    def _generate_ambiguous_classification(self, seed: int) -> tuple:
        """Generate ambiguous classification case."""
        random.seed(self.random_seed + seed)

        ambiguous_cases = [
            ("Business", "Conference Hotel", "Hotel stay for conference"),
            ("Personal", "Amazon", "Mixed personal and business items"),
            ("Investment", "Cryptocurrency Exchange", "Crypto purchase"),
            ("Gambling", "Gaming Platform", "In-game purchases"),
        ]

        return random.choice(ambiguous_cases)

    def _generate_fraud_pattern(self, risk_level: str, seed: int) -> tuple:
        """Generate fraud pattern for given risk level."""
        random.seed(self.random_seed + seed)

        patterns = {
            "LOW": (
                random.uniform(10, 200),
                "Regular Merchant",
                []
            ),
            "MEDIUM": (
                random.uniform(500, 2000),
                "Foreign Merchant",
                ["unusual_amount"]
            ),
            "HIGH": (
                random.uniform(5000, 10000),
                "Offshore Service",
                ["high_amount", "suspicious_merchant"]
            ),
            "CRITICAL": (
                random.uniform(20000, 100000),
                "Anonymous Crypto Exchange",
                ["very_high_amount", "anonymous_merchant", "crypto_related"]
            ),
        }

        return patterns[risk_level]

    def _generate_synthetic_fraud(self, seed: int) -> tuple:
        """Generate synthetic fraud pattern."""
        random.seed(self.random_seed + seed)

        patterns = [
            ("MEDIUM", random.uniform(1000, 5000), "Unknown Merchant", ["unusual_merchant"]),
            ("HIGH", random.uniform(10000, 20000), "Overseas Transfer", ["high_amount", "foreign"]),
            ("CRITICAL", random.uniform(50000, 100000), "Crypto ATM", ["very_high_amount", "crypto"]),
        ]

        return random.choice(patterns)

    def _get_mcc_for_risk(self, risk_level: str) -> str:
        """Get MCC code for risk level."""
        mcc_codes = {
            "LOW": "5411",  # Grocery stores
            "MEDIUM": "5999",  # Miscellaneous
            "HIGH": "6051",  # Crypto
            "CRITICAL": "7995",  # Gambling
        }
        return mcc_codes.get(risk_level, "0000")


def generate_full_dataset() -> dict:
    """
    Generate full evaluation dataset (REQ-EVAL-004).

    Returns:
        Dictionary with test cases for each tool
    """
    generator = DatasetGenerator(random_seed=42)

    return {
        "filter": generator.generate_filter_cases(100),
        "classify": generator.generate_classification_cases(100),
        "fraud": generator.generate_fraud_cases(100),
        "csv": generator.generate_csv_cases(100),
    }
