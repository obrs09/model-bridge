# src/model_bridge/cli.py
"""
CLI module for Model Bridge.

Provides command-line interface using Click and Rich.

Commands:
    mb scan     - Scan directories for models
    mb list     - List all registered models
    mb find     - Find models by query
    mb get      - Get path of a specific model
    mb config   - Manage configuration
    mb stats    - Show statistics
    mb clear    - Clear the registry
"""

import click
from pathlib import Path
from typing import Optional, List
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import ModelRegistry
from .config import ConfigManager
from .ranker import ModelRanker
from .utils import format_size

console = Console()


def get_registry() -> ModelRegistry:
    """Get or create the model registry singleton."""
    return ModelRegistry()


def get_config() -> ConfigManager:
    """Get or create the config manager singleton."""
    return ConfigManager()


# ============================================================
# Main CLI Group
# ============================================================

@click.group()
@click.version_option(version="0.2.0", prog_name="Model Bridge")
def main():
    """
    🌉 Model Bridge - Unified model manager for local AI models.
    
    Scan, index, and manage AI models from HuggingFace, GGUF, TensorRT, 
    ComfyUI, and more. Use fuzzy search to find models quickly.
    
    \b
    Quick Start:
      mb scan              # Scan for models
      mb find qwen 7b      # Find Qwen 7B models
      mb get llama         # Get path of best Llama match
    """
    pass


# ============================================================
# Scan Command
# ============================================================

@main.command()
@click.option(
    "--path", "-p",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Additional paths to scan (can be used multiple times)"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force full refresh (rebuild registry from scratch)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed scanning progress"
)
def scan(path: tuple, force: bool, verbose: bool):
    """
    Scan directories for models and update the registry.
    
    By default uses incremental mode (only adds new models).
    Use --force to rebuild registry from scratch.
    
    \b
    Examples:
      mb scan              # Incremental scan
      mb scan --force      # Full refresh
      mb scan -p D:/Models # Add new search path
      mb scan --verbose    # Show details
    """
    registry = get_registry()
    config = get_config()
    
    # Add any additional paths from CLI
    for p in path:
        config.add_search_path(str(p))
        if verbose:
            console.print(f"[dim]Added search path: {p}[/dim]")
    
    mode_str = "full refresh" if force else "incremental"
    console.print(Panel.fit(f"🔍 Scanning for models ({mode_str})...", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing scanner...", total=None)
        progress.update(task, description="Running scan strategies...")
        
        # Run the scan (returns statistics)
        scan_stats = registry.scan(force_refresh=force, verbose=verbose)
    
    # Show results
    reg_stats = registry.stats()
    
    console.print(f"[green]✅ Scan complete![/green]")
    console.print(f"   Scanned: [cyan]{scan_stats['scanned']}[/cyan] models")
    console.print(f"   Added: [green]{scan_stats['added']}[/green] new")
    console.print(f"   Skipped: [dim]{scan_stats['skipped']}[/dim] (already indexed)")
    console.print(f"   Total: [bold cyan]{scan_stats['total']}[/bold cyan] models")
    console.print(f"   Size: [cyan]{reg_stats.get('total_size_gb', 0):.2f} GB[/cyan]")
    
    if verbose:
        console.print("\n[bold]By Type:[/bold]")
        for model_type, count in reg_stats.get("by_type", {}).items():
            console.print(f"   {model_type}: {count}")


# ============================================================
# List Command
# ============================================================

@main.command("list")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["table", "json", "simple", "paths"]),
    default="table",
    help="Output format"
)
@click.option(
    "--type", "-t", "model_type",
    type=str,
    default=None,
    help="Filter by model type (e.g., gguf, hf_local)"
)
@click.option(
    "--limit", "-n",
    type=int,
    default=50,
    help="Maximum number of models to show"
)
def list_models(output_format: str, model_type: Optional[str], limit: int):
    """
    List all registered models.
    
    \b
    Examples:
      mb list
      mb list -t gguf
      mb list -f json
      mb list -f paths > models.txt
    """
    registry = get_registry()
    
    if registry.is_empty:
        console.print("[yellow]⚠️ Registry is empty.[/yellow]")
        console.print("Run [bold cyan]mb scan[/bold cyan] to discover models.")
        return
    
    # Get models
    if model_type:
        models = registry.get_by_type(model_type)
    else:
        # Get all models directly from registry
        models = registry.models[:limit * 2]  # Get more than needed
    
    models = models[:limit]
    
    if not models:
        console.print(f"[yellow]No models found{' of type ' + model_type if model_type else ''}.[/yellow]")
        return
    
    # Output based on format
    if output_format == "json":
        click.echo(json.dumps(models, indent=2))
    
    elif output_format == "simple":
        for model in models:
            console.print(f"[cyan]{model['id']}[/cyan] ({model.get('type', 'N/A')})")
    
    elif output_format == "paths":
        for model in models:
            click.echo(model.get('path', ''))
    
    else:  # table
        table = Table(title=f"📦 Registered Models ({len(models)} shown)")
        table.add_column("#", style="dim", width=4)
        table.add_column("ID", style="cyan", max_width=40)
        table.add_column("Type", style="green", width=12)
        table.add_column("Size", style="yellow", width=10)
        table.add_column("Path", style="dim", max_width=50)
        
        for i, model in enumerate(models, 1):
            path = model.get("path", "N/A")
            path_display = "..." + path[-47:] if len(path) > 50 else path
            
            table.add_row(
                str(i),
                model.get("id", "N/A"),
                model.get("type", "N/A"),
                format_size(model.get("size_bytes", 0)),
                path_display
            )
        
        console.print(table)
        
        total = registry.count
        if total > limit:
            console.print(f"[dim]Showing {limit} of {total} models. Use --limit to show more.[/dim]")


# ============================================================
# Find Command
# ============================================================

@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--top", "-n",
    type=int,
    default=10,
    help="Number of results to show"
)
@click.option(
    "--min-quant", "-q",
    type=str,
    default=None,
    help="Minimum quantization level (e.g., q4 for 'Q4 or above')"
)
@click.option(
    "--explain", "-e",
    is_flag=True,
    help="Show scoring breakdown for each result"
)
def find(query: tuple, top: int, min_quant: Optional[str], explain: bool):
    """
    Find models by fuzzy search query.
    
    Supports multiple search terms. Use size (7b, 72b), format (gguf),
    quantization (q4_k_m), and model type (instruct, base, moe).
    
    The --min-quant option filters by minimum precision level.
    
    \b
    Examples:
      mb find qwen
      mb find qwen 7b instruct
      mb find llama gguf q4_k_m
      mb find moe
      mb find qwen -q q4              # Q4 or above only
      mb find qwen 7b --min-quant q5  # Q5 or above
    """
    registry = get_registry()
    query_str = " ".join(query)
    
    if registry.is_empty:
        console.print("[yellow]⚠️ Registry is empty.[/yellow]")
        console.print("Run [bold cyan]mb scan[/bold cyan] first.")
        return
    
    console.print(f"[dim]Searching for: '{query_str}'[/dim]")
    if min_quant:
        console.print(f"[dim]Minimum quant: {min_quant.upper()} or above[/dim]")
    console.print()
    
    results = registry.find(query_str, top_k=top, min_quant=min_quant)
    
    # Normalize to list (find returns single dict when top_k=1)
    if results is None:
        results = []
    elif isinstance(results, dict):
        results = [results]
    
    if not results:
        console.print(f"[yellow]No models found matching '{query_str}'[/yellow]")
        
        # Suggest alternatives
        console.print("\n[dim]Tips:[/dim]")
        console.print("  • Try a simpler query (e.g., just the model name)")
        console.print("  • Check available models with: mb list")
        return
    
    console.print(f"[green]Found {len(results)} matching model(s):[/green]\n")
    
    for i, model in enumerate(results, 1):
        model_id = model.get("id", "N/A")
        model_type = model.get("type", "N/A")
        model_size = format_size(model.get("size_bytes", 0))
        model_path = model.get("path", "N/A")
        
        console.print(f"[bold cyan]{i}. {model_id}[/bold cyan]")
        console.print(f"   Type: {model_type} | Size: {model_size}")
        console.print(f"   Path: [dim]{model_path}[/dim]")
        
        if explain:
            explanation = ModelRanker.explain_score(query_str, model)
            if not explanation.get("rejected"):
                console.print(f"   Score: [green]{explanation['total']}[/green]")
                scores = explanation.get("scores", {})
                score_parts = [f"{k}: +{v}" for k, v in scores.items() if v > 0]
                if score_parts:
                    console.print(f"   [dim]({', '.join(score_parts)})[/dim]")
        
        console.print()


# ============================================================
# Get Command (Quick Path Lookup)
# ============================================================

@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--copy", "-c",
    is_flag=True,
    help="Copy path to clipboard (requires pyperclip)"
)
def get(query: tuple, copy: bool):
    """
    Get the path of a model (best match).
    
    Returns just the path, useful for scripting.
    
    \b
    Examples:
      mb get qwen
      model_path=$(mb get llama 7b)
    """
    registry = get_registry()
    query_str = " ".join(query)
    
    result = registry.find(query_str, top_k=1)
    
    if not result:
        console.print(f"[red]✗ Not found: {query_str}[/red]", err=True)
        raise SystemExit(1)
    
    model = result[0] if isinstance(result, list) else result
    path = model.get("path", "")
    
    if copy:
        try:
            import pyperclip
            pyperclip.copy(path)
            console.print(f"[green]✓ Copied to clipboard:[/green] {model.get('id', '')}")
        except ImportError:
            console.print("[yellow]Install pyperclip for clipboard support[/yellow]", err=True)
    
    # Always output the path (for piping)
    click.echo(path)


# ============================================================
# Config Command
# ============================================================

@main.group()
def config():
    """Manage Model Bridge configuration."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    cfg = get_config()
    
    console.print(Panel.fit("⚙️ Model Bridge Configuration", style="bold"))
    
    console.print("\n[bold cyan]Search Paths:[/bold cyan]")
    search_paths = cfg.get_search_paths()
    if search_paths:
        for i, path in enumerate(search_paths, 1):
            exists = "✓" if path.exists() else "✗"
            console.print(f"  {i}. [{exists}] {path}")
    else:
        console.print("  [dim](no custom paths configured)[/dim]")
    
    console.print("\n[bold cyan]Environment Variables:[/bold cyan]")
    env_vars = cfg.config.get("env_vars", {})
    if env_vars:
        for key, value in env_vars.items():
            console.print(f"  {key}={value}")
    else:
        console.print("  [dim](none configured)[/dim]")
    
    console.print(f"\n[bold cyan]Config File:[/bold cyan] {cfg.config_file_path}")


@config.command("add-path")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def config_add_path(path: Path):
    """Add a search path."""
    cfg = get_config()
    cfg.add_search_path(str(path.absolute()))
    console.print(f"[green]✓ Added:[/green] {path.absolute()}")


@config.command("set-hf-home")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def config_set_hf_home(path: Path):
    """Set HuggingFace cache directory."""
    cfg = get_config()
    cfg.set_hf_home(str(path.absolute()))
    console.print(f"[green]✓ HF_HOME set to:[/green] {path.absolute()}")


@config.command("reset")
@click.confirmation_option(prompt="Reset all configuration?")
def config_reset():
    """Reset configuration to defaults."""
    cfg = get_config()
    cfg.config = {"search_paths": [], "env_vars": {}}
    cfg.save()
    console.print("[green]✓ Configuration reset to defaults[/green]")


# ============================================================
# Stats Command
# ============================================================

@main.command()
def stats():
    """Show registry statistics."""
    registry = get_registry()
    
    if registry.is_empty:
        console.print("[yellow]Registry is empty. Run 'mb scan' first.[/yellow]")
        return
    
    stats_data = registry.stats()
    
    console.print(Panel.fit("📊 Registry Statistics", style="bold blue"))
    
    console.print(f"\n[bold]Total Models:[/bold] [cyan]{stats_data['total_models']}[/cyan]")
    total_gb = stats_data.get('total_size_gb', 0)
    console.print(f"[bold]Total Size:[/bold] [cyan]{total_gb:.2f} GB[/cyan]")
    
    # By type
    console.print("\n[bold]By Type:[/bold]")
    by_type = stats_data.get("by_type", {})
    if by_type:
        table = Table(show_header=False, box=None)
        table.add_column("Type", style="green", width=15)
        table.add_column("Count", style="cyan", width=10)
        
        for model_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            table.add_row(model_type, str(count))
        
        console.print(table)
    
    # Registry info
    console.print(f"\n[bold]Registry File:[/bold] [dim]{stats_data.get('registry_path', 'N/A')}[/dim]")
    console.print(f"[bold]Last Updated:[/bold] [dim]{stats_data.get('last_scan', 'Never')}[/dim]")


# ============================================================
# Clear Command
# ============================================================

@main.command()
@click.confirmation_option(prompt="Clear all models from registry?")
def clear():
    """Clear all models from the registry."""
    registry = get_registry()
    
    count = registry.count
    registry.clear()
    
    console.print(f"[green]✓ Cleared {count} models from registry[/green]")


# ============================================================
# Info Command
# ============================================================

@main.command()
@click.argument("model_query", nargs=-1, required=True)
def info(model_query: tuple):
    """Show detailed information about a model."""
    registry = get_registry()
    query_str = " ".join(model_query)
    
    result = registry.find(query_str, top_k=1)
    
    if not result:
        console.print(f"[red]Model not found: {query_str}[/red]")
        return
    
    model = result[0] if isinstance(result, list) else result
    
    console.print(Panel.fit(f"📦 {model.get('id', 'Unknown')}", style="bold cyan"))
    
    # Basic info
    console.print(f"\n[bold]Type:[/bold] {model.get('type', 'N/A')}")
    console.print(f"[bold]Size:[/bold] {format_size(model.get('size_bytes', 0))}")
    console.print(f"[bold]Path:[/bold] {model.get('path', 'N/A')}")
    
    # Parsed features
    features = ModelRanker.parse_features(model.get('id', '') + ' ' + Path(model.get('path', '')).name)
    
    console.print("\n[bold]Detected Features:[/bold]")
    console.print(f"  • Parameter Size: {features.get('size', 'Unknown')}B" if features.get('size') else "  • Parameter Size: Unknown")
    console.print(f"  • Quantization: {features.get('quant', 'N/A')}" if features.get('quant') else "  • Quantization: None (FP32/FP16)")
    console.print(f"  • Instruct/Chat: {'Yes' if features.get('is_instruct') else 'No'}")
    console.print(f"  • Base Model: {'Yes' if features.get('is_base') else 'No'}")
    console.print(f"  • MoE: {'Yes' if features.get('is_moe') else 'No'}")
    
    # Extra metadata if available
    if model.get('metadata'):
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in model['metadata'].items():
            console.print(f"  • {key}: {value}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
