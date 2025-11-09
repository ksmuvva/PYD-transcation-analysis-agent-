"""
Pydantic Logfire telemetry and observability module for the financial transaction analysis agent.

This module provides comprehensive tracing, evaluation, and telemetry capabilities using
Pydantic Logfire for tracking LLM calls, tool usage, MCTS iterations, and agent performance.
"""

import os
from typing import Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import logfire


@dataclass
class LogfireConfig:
    """Configuration for Pydantic Logfire telemetry."""

    enabled: bool = True
    project_name: str = "financial-transaction-agent"
    service_name: str = "transaction-analyzer"
    enable_pydantic_plugin: bool = True
    enable_mcts_telemetry: bool = True
    console_log: bool = True
    send_to_logfire: bool = True

    @classmethod
    def from_env(cls) -> "LogfireConfig":
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("LOGFIRE_ENABLED", "true").lower() == "true",
            project_name=os.getenv("LOGFIRE_PROJECT_NAME", "financial-transaction-agent"),
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "transaction-analyzer"),
            enable_pydantic_plugin=os.getenv(
                "LOGFIRE_PYDANTIC_PLUGIN", "true"
            ).lower() == "true",
            enable_mcts_telemetry=os.getenv(
                "LOGFIRE_MCTS_TELEMETRY", "true"
            ).lower() == "true",
            console_log=os.getenv("LOGFIRE_CONSOLE", "true").lower() == "true",
            send_to_logfire=os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "true").lower() == "true",
        )


class LogfireTelemetry:
    """
    Main telemetry class for managing Logfire observability.

    Provides:
    - Automatic LLM call tracing (OpenAI, Anthropic) via Pydantic AI integration
    - MCTS iteration tracking
    - Transaction analysis metrics
    - Custom span creation for domain-specific operations
    - Token usage and cost tracking
    """

    def __init__(self, config: Optional[LogfireConfig] = None):
        """
        Initialize Logfire telemetry.

        Args:
            config: Logfire configuration. If None, loads from environment.
        """
        self.config = config or LogfireConfig.from_env()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize Logfire and set up instrumentation."""
        if not self.config.enabled:
            print("Logfire telemetry is disabled")
            return

        if self._initialized:
            return

        try:
            # Configure Logfire
            logfire.configure(
                service_name=self.config.service_name,
                console=logfire.ConsoleOptions(colors='auto') if self.config.console_log else False,
                send_to_logfire=self.config.send_to_logfire,
            )

            # Instrument Pydantic AI (automatic LLM tracing)
            if self.config.enable_pydantic_plugin:
                logfire.instrument_pydantic()
                print("✅ Pydantic instrumentation enabled")

            # Instrument OpenAI
            try:
                logfire.instrument_openai()
                print("✅ OpenAI instrumentation enabled")
            except Exception as e:
                print(f"⚠️  OpenAI instrumentation unavailable: {e}")

            # Instrument Anthropic
            try:
                logfire.instrument_anthropic()
                print("✅ Anthropic instrumentation enabled")
            except Exception as e:
                print(f"⚠️  Anthropic instrumentation unavailable: {e}")

            self._initialized = True
            print(f"\n🔥 Logfire telemetry initialized for project: {self.config.project_name}")
            if self.config.send_to_logfire:
                print(f"   View traces at: https://logfire.pydantic.dev/{self.config.project_name}\n")
            else:
                print("   Traces will only be logged to console\n")

        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Logfire: {e}")
            print("   Continuing without telemetry...")

    @contextmanager
    def span(
        self,
        name: str,
        **attributes: Any,
    ):
        """
        Create a custom span for tracking operations.

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

        with logfire.span(name, **attributes) as span:
            yield span

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
