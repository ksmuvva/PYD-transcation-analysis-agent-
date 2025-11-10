"""
CLI interface for the financial transaction analysis agent.

Provides command-line interface for running transaction analysis.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.agent import run_analysis
from src.config import ConfigManager
from src.csv_processor import CSVProcessor
from src.telemetry import initialize_telemetry, LogfireConfig

app = typer.Typer(
    name="financial-agent",
    help="AI-powered financial transaction analysis with MCTS reasoning",
)
console = Console()


@app.command()
def analyze(
    csv_file: Path = typer.Argument(..., help="Path to input CSV file with transactions"),
    output: Path = typer.Option(
        "enhanced_transactions.csv",
        "--output",
        "-o",
        help="Path for output CSV file",
    ),
    llm_provider: str = typer.Option(
        "openai",
        "--llm-provider",
        "-p",
        help="LLM provider (openai, anthropic)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (must be a reasoning model like o1-mini)",
    ),
    threshold: float = typer.Option(
        250.0,
        "--threshold",
        "-t",
        help="Transaction threshold amount",
    ),
    currency: str = typer.Option(
        "GBP",
        "--currency",
        "-c",
        help="Base currency for filtering (GBP, USD, EUR, etc.)",
    ),
    mcts_iterations: int = typer.Option(
        100,
        "--mcts-iterations",
        "-i",
        help="Number of MCTS iterations per transaction",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    enable_telemetry: bool = typer.Option(
        True,
        "--telemetry/--no-telemetry",
        help="Enable Pydantic Logfire telemetry and observability",
    ),
) -> None:
    """
    Analyze financial transactions for classification and fraud detection.

    This command processes a CSV file of transactions and:
    1. Filters transactions above the specified threshold
    2. Classifies each transaction using MCTS reasoning
    3. Detects fraudulent transactions using MCTS reasoning
    4. Generates an enhanced CSV with analysis results

    Example:
        financial-agent analyze transactions.csv --model o1-mini --threshold 250
    """
    console.print("\n[bold blue]Financial Transaction Analysis Agent[/bold blue]")
    console.print("[dim]Powered by Pydantic AI + MCTS Reasoning[/dim]\n")

    # Initialize Logfire telemetry
    telemetry = None
    if enable_telemetry:
        try:
            logfire_config = LogfireConfig.from_env()
            telemetry = initialize_telemetry(logfire_config)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to initialize Logfire telemetry: {e}[/yellow]")
            console.print("[yellow]Continuing without telemetry...[/yellow]")

    try:
        # Step 1: Validate inputs
        if verbose:
            console.print("[yellow]Validating inputs...[/yellow]")

        if not csv_file.exists():
            console.print(f"[red]Error: CSV file not found: {csv_file}[/red]")
            raise typer.Exit(1)

        if not model:
            console.print("[red]Error: Model name required. Use --model flag.[/red]")
            console.print("[dim]Example: --model o1-mini[/dim]")
            raise typer.Exit(1)

        # Step 2: Load configuration
        if verbose:
            console.print("[yellow]Loading configuration...[/yellow]")
            console.print(f"  Provider: {llm_provider}")
            console.print(f"  Model: {model}")
            console.print(f"  Threshold: {threshold} {currency}")
            console.print(f"  MCTS Iterations: {mcts_iterations}")

        try:
            config = ConfigManager.load_from_env(
                provider=llm_provider,
                model=model,
                api_key=None,  # Always use environment variable for security
                threshold=threshold,
                currency=currency,
                mcts_iterations=mcts_iterations,
            )
        except ValueError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(1)

        # Step 3: Load CSV
        console.print(f"\n[cyan]Loading CSV file: {csv_file}[/cyan]")

        try:
            df = CSVProcessor.load_csv(csv_file)
            console.print(f"[green]✓[/green] Loaded {len(df)} transactions")
        except Exception as e:
            console.print(f"[red]Failed to load CSV: {e}[/red]")
            raise typer.Exit(1)

        # Step 4: Validate schema
        if verbose:
            console.print("\n[yellow]Validating CSV schema...[/yellow]")

        errors = CSVProcessor.validate_schema(df)
        if errors:
            console.print("[red]CSV validation errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)

        console.print("[green]✓[/green] CSV schema valid")

        # Step 5: Run analysis
        console.print("\n[bold cyan]Running Analysis...[/bold cyan]\n")

        progress_messages = []

        def progress_callback(message: str) -> None:
            """Callback for progress updates."""
            progress_messages.append(message)
            console.print(f"[dim]{message}[/dim]")

        try:
            report = run_analysis(
                df=df,
                config=config,
                output_path=output,
                progress_callback=progress_callback if verbose else None,
            )
        except Exception as e:
            console.print(f"\n[red]Analysis failed: {e}[/red]")
            if verbose:
                import traceback
                console.print(f"\n[red]{traceback.format_exc()}[/red]")
            raise typer.Exit(1)

        # Step 6: Display results
        console.print("\n[bold green]✓ Analysis Complete![/bold green]\n")

        # Display summary table
        table = Table(title="Processing Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Transactions Analyzed", str(report.total_transactions_analyzed))
        table.add_row("Transactions Above Threshold", str(report.transactions_above_threshold))
        table.add_row(
            "High Risk Transactions",
            f"[red]{report.high_risk_transactions}[/red]"
            if report.high_risk_transactions > 0
            else "0",
        )
        table.add_row(
            "Critical Risk Transactions",
            f"[bold red]{report.critical_risk_transactions}[/bold red]"
            if report.critical_risk_transactions > 0
            else "0",
        )
        table.add_row("Processing Time", f"{report.processing_time_seconds:.2f}s")
        table.add_row("LLM Provider", report.llm_provider)
        table.add_row("Model Used", report.model_used)
        table.add_row("Total MCTS Iterations", str(report.mcts_iterations_total))

        console.print(table)

        # Output file info
        console.print(f"\n[bold green]Enhanced CSV saved to:[/bold green] {output.absolute()}")

        if report.high_risk_transactions > 0:
            console.print(
                f"\n[yellow]⚠ Warning: {report.high_risk_transactions} high-risk "
                f"transactions detected. Please review the output file.[/yellow]"
            )

        if report.critical_risk_transactions > 0:
            console.print(
                f"[bold red]⚠⚠ CRITICAL: {report.critical_risk_transactions} critical-risk "
                f"transactions require immediate attention![/bold red]"
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(1)
    finally:
        # Shutdown telemetry
        if telemetry:
            telemetry.shutdown()


@app.command()
def validate(
    csv_file: Path = typer.Argument(..., help="Path to CSV file to validate"),
) -> None:
    """
    Validate CSV file format without running analysis.

    Checks that the CSV file has the required columns and valid data.
    """
    console.print(f"\n[cyan]Validating CSV file: {csv_file}[/cyan]\n")

    try:
        # Load CSV
        df = CSVProcessor.load_csv(csv_file)
        console.print(f"[green]✓[/green] File loaded successfully ({len(df)} rows)")

        # Validate schema
        errors = CSVProcessor.validate_schema(df)

        if errors:
            console.print("\n[red]Validation Errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)
        else:
            console.print("[green]✓[/green] Schema validation passed")

            # Show column info
            console.print("\n[cyan]Columns found:[/cyan]")
            for col in df.columns:
                console.print(f"  • {col}")

            console.print("\n[bold green]✓ CSV file is valid![/bold green]")

    except Exception as e:
        console.print(f"\n[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models() -> None:
    """
    List available reasoning models for each LLM provider.

    Shows the whitelist of models that can be used with this agent.
    """
    console.print("\n[bold cyan]Available Reasoning Models[/bold cyan]\n")

    for provider, model_list in ConfigManager.REASONING_MODELS.items():
        console.print(f"[yellow]{provider.upper()}:[/yellow]")
        for model in model_list:
            console.print(f"  • {model}")
        console.print()

    console.print("[dim]Note: Only reasoning models are supported.[/dim]")


if __name__ == "__main__":
    app()
