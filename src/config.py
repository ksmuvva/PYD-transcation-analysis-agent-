"""
Configuration management for the financial transaction analysis agent.

Handles LLM provider configuration, MCTS parameters, and validation.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

from src.models import Currency

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """
    LLM provider configuration.

    Attributes:
        provider: Provider name (openai, anthropic, etc.)
        model: Model name (must be a reasoning model)
        api_key: API key for the provider
        temperature: Sampling temperature (default 0.0 for reasoning models)
        max_tokens: Maximum tokens in response
    """

    provider: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 4000


@dataclass
class MCTSConfig:
    """
    MCTS reasoning engine configuration.

    Attributes:
        iterations: Number of MCTS iterations per search
        exploration_constant: UCB1 exploration constant (√2 is common)
        max_depth: Maximum tree depth
        simulation_budget: Number of simulations per node
    """

    iterations: int = 100
    exploration_constant: float = 1.414  # sqrt(2)
    max_depth: int = 5
    simulation_budget: int = 10


@dataclass
class AgentConfig:
    """
    Complete agent configuration.

    Combines LLM and MCTS configurations with transaction processing parameters.
    """

    llm: LLMConfig
    mcts: MCTSConfig
    threshold_amount: float = 250.0
    base_currency: Currency = Currency.GBP


class ConfigManager:
    """Manages configuration validation and LLM client creation."""

    # Whitelist of reasoning models
    REASONING_MODELS = {
        "openai": [
            "o1",
            "o1-preview",
            "o1-mini",
            "o3-mini",
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-5-20250929",
        ],
    }

    @staticmethod
    def validate_reasoning_model(provider: str, model: str) -> bool:
        """
        Validate that the model is a reasoning model.

        Args:
            provider: LLM provider name
            model: Model name

        Returns:
            True if model is a reasoning model

        Raises:
            ValueError: If model is not a reasoning model
        """
        provider = provider.lower()

        if provider not in ConfigManager.REASONING_MODELS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {list(ConfigManager.REASONING_MODELS.keys())}"
            )

        allowed_models = ConfigManager.REASONING_MODELS[provider]

        # Check if model matches any allowed model (exact or prefix match)
        if not any(model.startswith(allowed) for allowed in allowed_models):
            raise ValueError(
                f"Model '{model}' is not a reasoning model for provider '{provider}'. "
                f"Allowed reasoning models: {allowed_models}"
            )

        return True

    @staticmethod
    def create_llm_client(config: LLMConfig) -> Model:
        """
        Create a Pydantic AI compatible LLM client.

        Args:
            config: LLM configuration

        Returns:
            Pydantic AI Model instance

        Raises:
            ValueError: If provider is not supported or API key is invalid
        """
        # Validate reasoning model
        ConfigManager.validate_reasoning_model(config.provider, config.model)

        # Validate API key
        if not config.api_key or config.api_key.startswith("your-"):
            raise ValueError(
                f"Invalid API key for {config.provider}. "
                f"Please set {config.provider.upper()}_API_KEY environment variable "
                f"or provide via --api-key flag."
            )

        # Create client based on provider
        if config.provider.lower() == "openai":
            return OpenAIModel(
                config.model,
                api_key=config.api_key,
            )
        elif config.provider.lower() == "anthropic":
            # Set API key in environment for AnthropicModel
            # (new pydantic-ai API reads from environment)
            os.environ['ANTHROPIC_API_KEY'] = config.api_key
            return AnthropicModel(config.model)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    @staticmethod
    def load_from_env(
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        threshold: float | None = None,
        currency: str | None = None,
        mcts_iterations: int | None = None,
    ) -> AgentConfig:
        """
        Load configuration from environment variables with optional overrides.

        Args:
            provider: LLM provider (overrides DEFAULT_LLM_PROVIDER)
            model: Model name (overrides DEFAULT_MODEL)
            api_key: API key (overrides <PROVIDER>_API_KEY)
            threshold: Transaction threshold (overrides DEFAULT_THRESHOLD)
            currency: Base currency (overrides DEFAULT_CURRENCY)
            mcts_iterations: MCTS iterations (overrides DEFAULT_MCTS_ITERATIONS)

        Returns:
            Complete AgentConfig

        Raises:
            ValueError: If required configuration is missing
        """
        # LLM Configuration
        llm_provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        llm_model = model or os.getenv("DEFAULT_MODEL")

        if not llm_model:
            raise ValueError("Model name required. Set DEFAULT_MODEL env var or use --model flag.")

        # Get API key from provider-specific env var or parameter
        if api_key:
            llm_api_key = api_key
        else:
            llm_api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")

        llm_config = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        # MCTS Configuration
        mcts_config = MCTSConfig(
            iterations=mcts_iterations
            or int(os.getenv("DEFAULT_MCTS_ITERATIONS", "100")),
            exploration_constant=float(os.getenv("MCTS_EXPLORATION_CONSTANT", "1.414")),
            max_depth=int(os.getenv("MCTS_MAX_DEPTH", "5")),
            simulation_budget=int(os.getenv("MCTS_SIMULATION_BUDGET", "10")),
        )

        # Agent Configuration
        threshold_amount = threshold or float(os.getenv("DEFAULT_THRESHOLD", "250.0"))
        base_currency_str = currency or os.getenv("DEFAULT_CURRENCY", "GBP")

        try:
            base_currency = Currency(base_currency_str)
        except ValueError:
            raise ValueError(
                f"Invalid currency: {base_currency_str}. "
                f"Supported currencies: {[c.value for c in Currency]}"
            )

        return AgentConfig(
            llm=llm_config,
            mcts=mcts_config,
            threshold_amount=threshold_amount,
            base_currency=base_currency,
        )
