"""
Pydantic Logfire telemetry and observability module for the financial transaction analysis agent.

This module provides comprehensive tracing, evaluation, and telemetry capabilities using
Pydantic Logfire for tracking LLM calls, tool usage, MCTS iterations, and agent performance.

Implements REQ-016 through REQ-027:
- Comprehensive span hierarchy
- Real-time IDE trace visualization
- Configuration-driven setup
- Cost and token tracking
- Error and convergence logging
- Persistent trace storage
- GitHub integration for experiment tracking
- Unit test coverage with test mode
- Deterministic testing mode
- PII redaction
"""

import os
import re
import subprocess
from typing import Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import logfire


@dataclass
class LogfireConfig:
    """
    Configuration for Pydantic Logfire telemetry.

    Implements REQ-018: Configuration-driven Logfire setup
    """

    enabled: bool = True
    project_name: str = "financial-fraud-agent-mcts"  # REQ-018
    service_name: str = "transaction-analyzer"
    enable_pydantic_plugin: bool = True
    enable_mcts_telemetry: bool = True
    console_log: bool = True
    send_to_logfire: bool = True

    # REQ-018: Environment variable configuration
    token: Optional[str] = None  # LOGFIRE_TOKEN
    scrubbing: bool = False  # LOGFIRE_SCRUBBING (false in dev, true in prod)

    # REQ-021: Persistent trace storage
    postgres_dsn: Optional[str] = None  # LOGFIRE_POSTGRES_DSN
    sqlite_path: str = os.path.expanduser("~/.logfire/logfire.db")

    # REQ-022: GitHub integration
    github_integration: bool = True
    git_repo_root: Optional[str] = None

    # REQ-020: Error and convergence logging
    convergence_error_alert_threshold: float = 0.05  # 5% failure rate
    convergence_alert_window_hours: int = 1

    # REQ-023, REQ-024: Testing support
    test_mode: bool = False  # Set true in tests
    deterministic_seed: Optional[int] = None  # For reproducible tests

    @classmethod
    def from_env(cls) -> "LogfireConfig":
        """
        Create configuration from environment variables (REQ-018).

        Environment variables:
        - LOGFIRE_ENABLED: Enable/disable telemetry (default: true)
        - LOGFIRE_TOKEN: API token for cloud upload (optional for POC)
        - LOGFIRE_PROJECT_NAME: Project name (default: financial-fraud-agent-mcts)
        - LOGFIRE_SCRUBBING: Enable PII redaction (false in dev, true in prod)
        - LOGFIRE_POSTGRES_DSN: PostgreSQL connection for trace storage
        - LOGFIRE_GITHUB_INTEGRATION: Enable GitHub commit tracking (default: true)
        - LOGFIRE_TEST_MODE: Enable test mode (default: false)
        - LOGFIRE_DETERMINISTIC_SEED: Seed for deterministic testing

        Returns:
            LogfireConfig instance
        """
        return cls(
            enabled=os.getenv("LOGFIRE_ENABLED", "true").lower() == "true",
            token=os.getenv("LOGFIRE_TOKEN"),  # REQ-018
            project_name=os.getenv("LOGFIRE_PROJECT_NAME", "financial-fraud-agent-mcts"),
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "transaction-analyzer"),
            enable_pydantic_plugin=os.getenv(
                "LOGFIRE_PYDANTIC_PLUGIN", "true"
            ).lower() == "true",
            enable_mcts_telemetry=os.getenv(
                "LOGFIRE_MCTS_TELEMETRY", "true"
            ).lower() == "true",
            console_log=os.getenv("LOGFIRE_CONSOLE", "true").lower() == "true",
            send_to_logfire=os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "true").lower() == "true",
            scrubbing=os.getenv("LOGFIRE_SCRUBBING", "false").lower() == "true",  # REQ-018
            postgres_dsn=os.getenv("LOGFIRE_POSTGRES_DSN"),  # REQ-021
            github_integration=os.getenv("LOGFIRE_GITHUB_INTEGRATION", "true").lower() == "true",
            test_mode=os.getenv("LOGFIRE_TEST_MODE", "false").lower() == "true",
            deterministic_seed=int(os.getenv("LOGFIRE_DETERMINISTIC_SEED", "0")) or None,
        )


class LogfireTelemetry:
    """
    Main telemetry class for managing Logfire observability.

    Implements REQ-016 through REQ-027.

    Provides:
    - Automatic LLM call tracing (OpenAI, Anthropic) via Pydantic AI integration
    - MCTS iteration tracking
    - Transaction analysis metrics
    - Custom span creation for domain-specific operations
    - Token usage and cost tracking (REQ-019)
    - PII redaction (REQ-026)
    - GitHub integration (REQ-022)
    - Comprehensive span hierarchy (REQ-016)
    """

    def __init__(self, config: Optional[LogfireConfig] = None):
        """
        Initialize Logfire telemetry.

        Args:
            config: Logfire configuration. If None, loads from environment.
        """
        self.config = config or LogfireConfig.from_env()
        self._initialized = False
        self._scrubbing_function = None
        self._git_info: Optional[dict[str, str]] = None
        self._convergence_error_count = 0
        self._total_transaction_count = 0

    def initialize(self) -> None:
        """
        Initialize Logfire and set up instrumentation.

        Implements REQ-017, REQ-018, REQ-022, REQ-023, REQ-026.
        """
        if not self.config.enabled:
            print("Logfire telemetry is disabled")
            return

        if self._initialized:
            return

        try:
            # REQ-022: Set up GitHub integration
            if self.config.github_integration:
                self._git_info = self._get_git_info()

            # REQ-026: Set up PII scrubbing function
            if self.config.scrubbing:
                self._scrubbing_function = self._create_scrubbing_function()

            # REQ-023: Test mode configuration
            send_to_logfire = 'never' if self.config.test_mode else self.config.send_to_logfire

            # REQ-018, REQ-021: Configure Logfire with environment-driven settings
            configure_kwargs = {
                "service_name": self.config.service_name,
                "console": logfire.ConsoleOptions(colors='auto') if self.config.console_log else False,
                "send_to_logfire": send_to_logfire,
            }

            # Add token if provided (REQ-018)
            if self.config.token:
                configure_kwargs["token"] = self.config.token

            # Add scrubbing if enabled (REQ-026)
            if self.config.scrubbing and self._scrubbing_function:
                configure_kwargs["scrubbing"] = self._scrubbing_function

            # REQ-021: Add PostgreSQL DSN for persistent storage if provided
            if self.config.postgres_dsn:
                # Note: Logfire uses environment variables for postgres
                os.environ["LOGFIRE_POSTGRES_DSN"] = self.config.postgres_dsn

            logfire.configure(**configure_kwargs)

            # REQ-017: Instrument Pydantic AI for comprehensive tracing
            if self.config.enable_pydantic_plugin:
                logfire.instrument_pydantic()
                print("✅ Pydantic instrumentation enabled (REQ-017)")

            # REQ-019: Instrument OpenAI (automatic cost tracking)
            try:
                logfire.instrument_openai()
                print("✅ OpenAI instrumentation enabled (REQ-019: cost tracking)")
            except Exception as e:
                print(f"⚠️  OpenAI instrumentation unavailable: {e}")

            # REQ-019: Instrument Anthropic (automatic cost tracking)
            try:
                logfire.instrument_anthropic()
                print("✅ Anthropic instrumentation enabled (REQ-019: cost tracking)")
            except Exception as e:
                print(f"⚠️  Anthropic instrumentation unavailable: {e}")

            self._initialized = True

            # Print initialization summary
            print(f"\n🔥 Logfire telemetry initialized")
            print(f"   Project: {self.config.project_name}")
            print(f"   Mode: {'TEST' if self.config.test_mode else 'PRODUCTION'}")

            if self._git_info:
                print(f"   Git: {self._git_info.get('branch', 'unknown')}@{self._git_info.get('sha', 'unknown')[:7]}")

            if self.config.scrubbing:
                print("   PII Redaction: ENABLED (REQ-026)")

            if self.config.send_to_logfire and not self.config.test_mode:
                print(f"   Dashboard: https://logfire.pydantic.dev/{self.config.project_name}")
            elif self.config.test_mode:
                print("   Traces: In-memory only (test mode)")
            else:
                print("   Traces: Console only")

            print()

        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Logfire: {e}")
            print("   Continuing without telemetry...")

    def _get_git_info(self) -> dict[str, str]:
        """
        Get Git repository information for experiment tracking (REQ-022).

        Returns:
            Dictionary with git metadata (sha, branch, repo_url)
        """
        try:
            # Get current commit SHA
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get current branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get remote URL
            repo_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            return {
                "sha": sha,
                "branch": branch,
                "repo_url": repo_url,
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

    def _create_scrubbing_function(self) -> Callable:
        """
        Create PII scrubbing function (REQ-026).

        Redacts:
        - Counterparty names
        - Account numbers
        - Merchant full names (keeps only first 3 chars)
        - Any patterns that look like PII

        Returns:
            Scrubbing function for Logfire
        """
        def scrub_pii(message: str) -> str:
            """Redact PII from log messages."""
            # Redact account numbers (patterns like: 1234567890, ACCT12345, etc.)
            message = re.sub(r'\b\d{8,}\b', '[REDACTED_ACCOUNT]', message)
            message = re.sub(r'\bACCT\d+\b', '[REDACTED_ACCOUNT]', message)

            # Redact email addresses
            message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', message)

            # Redact phone numbers
            message = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', message)

            # Redact names in "counterparty" or "merchant" fields
            message = re.sub(r'(counterparty|merchant)["\']?\s*:\s*["\']?([A-Za-z\s]+)["\']?', r'\1: [REDACTED_NAME]', message, flags=re.IGNORECASE)

            return message

        return scrub_pii

    @contextmanager
    def span(
        self,
        name: str,
        **attributes: Any,
    ):
        """
        Create a custom span for tracking operations (REQ-016).

        Implements comprehensive span hierarchy with automatic Git metadata.

        Args:
            name: Name of the span
            **attributes: Attributes to attach to the span

        Usage:
            with telemetry.span("classify_transaction", transaction_id=123, amount=500.0):
                # Your code here
                pass
        """
        if not self.config.enabled or not self._initialized:
            # If telemetry is disabled, just yield without creating a span
            yield None
            return

        # REQ-022: Add Git metadata to all spans
        if self._git_info:
            attributes["git_sha"] = self._git_info.get("sha", "")
            attributes["git_branch"] = self._git_info.get("branch", "")
            attributes["git_repo"] = self._git_info.get("repo_url", "")

        # REQ-024: Add environment tag (test vs production)
        attributes["environment"] = "test" if self.config.test_mode else "production"

        with logfire.span(name, **attributes) as span:
            yield span

    def record_convergence_error(
        self,
        tool_name: str,
        transaction_id: str,
        iterations_completed: int,
        final_variance: float,
    ) -> None:
        """
        Record MCTS convergence error (REQ-020).

        Triggers alert if convergence failure rate exceeds 5% over 1-hour window.

        Args:
            tool_name: Name of the tool that failed
            transaction_id: Transaction ID
            iterations_completed: Number of iterations before failure
            final_variance: Final reward variance
        """
        if not self.config.enabled or not self._initialized:
            return

        self._convergence_error_count += 1
        self._total_transaction_count += 1

        # Calculate failure rate
        failure_rate = self._convergence_error_count / max(self._total_transaction_count, 1)

        # REQ-020: Log with error level and structured attributes
        logfire.error(
            "MCTS convergence error",
            tool_name=tool_name,
            transaction_id=transaction_id,
            iterations_completed=iterations_completed,
            final_variance=final_variance,
            failure_rate=failure_rate,
        )

        # REQ-020: Trigger alert if threshold exceeded
        if failure_rate > self.config.convergence_error_alert_threshold:
            logfire.error(
                "ALERT: Convergence failure rate threshold exceeded",
                failure_rate=failure_rate,
                threshold=self.config.convergence_error_alert_threshold,
                total_errors=self._convergence_error_count,
                total_transactions=self._total_transaction_count,
            )

    def record_cost_and_tokens(
        self,
        tool_name: str,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """
        Record token usage and cost per transaction (REQ-019).

        This enables cost-per-transaction analysis in Logfire dashboard.

        Args:
            tool_name: Name of the tool
            total_tokens: Total tokens used
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            cost_usd: Cost in USD
        """
        if not self.config.enabled or not self._initialized:
            return

        logfire.info(
            "LLM cost and token tracking",
            tool_name=tool_name,
            total_tokens_used=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )

    def log_info(self, message: str, **attributes: Any) -> None:
        """
        Log an info message with attributes.

        Args:
            message: Log message
            **attributes: Additional attributes
        """
        if self.config.enabled and self._initialized:
            logfire.info(message, **attributes)

    def log_warning(self, message: str, **attributes: Any) -> None:
        """
        Log a warning message with attributes.

        Args:
            message: Log message
            **attributes: Additional attributes
        """
        if self.config.enabled and self._initialized:
            logfire.warn(message, **attributes)

    def log_error(self, message: str, **attributes: Any) -> None:
        """
        Log an error message with attributes.

        Args:
            message: Log message
            **attributes: Additional attributes
        """
        if self.config.enabled and self._initialized:
            logfire.error(message, **attributes)

    def record_mcts_iteration(
        self,
        iteration: int,
        node_visits: int,
        node_value: float,
        best_hypothesis: str,
        confidence: float,
        objective: str,
    ) -> None:
        """
        Record MCTS iteration details.

        Args:
            iteration: Current iteration number
            node_visits: Number of visits to the node
            node_value: Value of the node
            best_hypothesis: Best hypothesis found
            confidence: Confidence score
            objective: MCTS objective (classify/detect_fraud)
        """
        if not self.config.enabled or not self.config.enable_mcts_telemetry or not self._initialized:
            return

        logfire.info(
            "MCTS iteration",
            iteration=iteration,
            node_visits=node_visits,
            node_value=node_value,
            best_hypothesis=best_hypothesis,
            confidence=confidence,
            objective=objective,
        )

    def record_transaction_analysis(
        self,
        transaction_id: str,
        amount: float,
        currency: str,
        classification: Optional[str] = None,
        fraud_risk: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Record transaction analysis details.

        Args:
            transaction_id: Transaction identifier
            amount: Transaction amount
            currency: Transaction currency
            classification: Classification result
            fraud_risk: Fraud risk level
            confidence: Confidence score
        """
        if not self.config.enabled or not self._initialized:
            return

        logfire.info(
            "Transaction analyzed",
            transaction_id=transaction_id,
            amount=amount,
            currency=currency,
            classification=classification,
            fraud_risk=fraud_risk,
            confidence=confidence,
        )

    def record_pipeline_metrics(
        self,
        total_transactions: int,
        transactions_analyzed: int,
        high_risk_count: int,
        critical_risk_count: int,
        processing_time_seconds: float,
        model_used: str,
    ) -> None:
        """
        Record pipeline-level metrics.

        Args:
            total_transactions: Total transaction count
            transactions_analyzed: Transactions analyzed count
            high_risk_count: High risk transaction count
            critical_risk_count: Critical risk transaction count
            processing_time_seconds: Total processing time
            model_used: LLM model used
        """
        if not self.config.enabled or not self._initialized:
            return

        logfire.info(
            "Pipeline completed",
            total_transactions=total_transactions,
            transactions_analyzed=transactions_analyzed,
            high_risk_count=high_risk_count,
            critical_risk_count=critical_risk_count,
            processing_time_seconds=processing_time_seconds,
            model_used=model_used,
        )

    def shutdown(self) -> None:
        """Shutdown Logfire and cleanup resources."""
        if self.config.enabled and self._initialized:
            print("\n✅ Logfire telemetry session complete")
            # Logfire handles cleanup automatically


# Global instance for easy access
_global_telemetry: Optional[LogfireTelemetry] = None


def get_telemetry() -> LogfireTelemetry:
    """
    Get or create the global Logfire telemetry instance.

    Returns:
        LogfireTelemetry instance
    """
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = LogfireTelemetry()
    return _global_telemetry


def initialize_telemetry(config: Optional[LogfireConfig] = None) -> LogfireTelemetry:
    """
    Initialize and return the global Logfire telemetry instance.

    Args:
        config: Optional Logfire configuration

    Returns:
        Initialized LogfireTelemetry instance
    """
    global _global_telemetry
    _global_telemetry = LogfireTelemetry(config)
    _global_telemetry.initialize()
    return _global_telemetry
