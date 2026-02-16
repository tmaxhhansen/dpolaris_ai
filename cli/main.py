#!/usr/bin/env python3
"""
dPolaris CLI - Command Line Interface

Usage:
    dpolaris start          Start the daemon
    dpolaris stop           Stop the daemon
    dpolaris status         Check status
    dpolaris chat           Interactive chat
    dpolaris scout          Scan for opportunities
    dpolaris analyze SYMBOL Analyze a symbol
    dpolaris predict SYMBOL ML prediction
    dpolaris train SYMBOL   Train model
    dpolaris server         Start API server
    dpolaris orchestrator   Start self-healing orchestrator daemon
    dpolaris backup         Create backup
    dpolaris setup          Initial setup
"""

import asyncio
import logging
import os
import sys
import signal
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_components():
    """Lazy load components"""
    from core.config import Config, get_config
    from core.database import Database
    from core.memory import DPolarisMemory
    from core.ai import DPolarisAI

    config = get_config()
    db = Database()
    memory = DPolarisMemory(db)
    ai = DPolarisAI(config, db, memory)

    return config, db, memory, ai


@click.group()
@click.version_option(version="0.1.0", prog_name="dPolaris")
def cli():
    """dPolaris AI - Trading Intelligence System"""
    pass


# ==================== Daemon Commands ====================

@cli.command()
def start():
    """Start the dPolaris daemon in background"""
    import subprocess

    config, _, _, _ = get_components()
    pid_file = config.data_dir / "daemon.pid"

    # Check if already running
    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
            console.print("[yellow]Daemon already running[/yellow]")
            return
        except OSError:
            pid_file.unlink()

    # Start daemon in background
    console.print("Starting dPolaris daemon...")

    log_file = config.data_dir / "logs" / "daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [sys.executable, "-m", "daemon"],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=Path(__file__).parent.parent,
    )

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(process.pid))

    console.print(f"[green]dPolaris daemon started (PID: {process.pid})[/green]")
    console.print(f"Logs: {log_file}")


@cli.command()
def stop():
    """Stop the dPolaris daemon"""
    config, _, _, _ = get_components()
    pid_file = config.data_dir / "daemon.pid"

    if not pid_file.exists():
        console.print("[yellow]Daemon not running[/yellow]")
        return

    pid = int(pid_file.read_text())

    try:
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
        console.print("[green]dPolaris daemon stopped[/green]")
    except OSError:
        pid_file.unlink()
        console.print("[yellow]Daemon was not running[/yellow]")


@cli.command()
def status():
    """Check daemon and system status"""
    config, db, _, _ = get_components()

    # Check daemon
    pid_file = config.data_dir / "daemon.pid"
    daemon_running = False

    if pid_file.exists():
        pid = int(pid_file.read_text())
        try:
            os.kill(pid, 0)
            daemon_running = True
        except OSError:
            pass

    # Get portfolio
    portfolio = db.get_latest_portfolio()

    # Build status table
    table = Table(title="dPolaris Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row(
        "Daemon",
        "[green]Running[/green]" if daemon_running else "[red]Stopped[/red]"
    )

    if portfolio:
        table.add_row("Portfolio Value", f"${portfolio.get('total_value', 0):,.2f}")
        table.add_row("Daily P/L", f"${portfolio.get('daily_pnl', 0):+,.2f}")
        table.add_row("Goal Progress", f"{portfolio.get('goal_progress', 0):.1f}%")
    else:
        table.add_row("Portfolio", "[yellow]Not initialized[/yellow]")

    table.add_row("Data Dir", str(config.data_dir))
    table.add_row("Goal", f"${config.goal.target:,.0f}")

    console.print(table)


# ==================== Interactive Commands ====================

@cli.command()
def chat():
    """Interactive chat with dPolaris AI"""
    _, _, _, ai = get_components()

    console.print(Panel(
        "dPolaris AI - Interactive Chat\n"
        "Commands: @scout, @analyze SYMBOL, @predict SYMBOL, @risk, @regime\n"
        "Type 'exit' or 'quit' to end session",
        title="dPolaris",
        border_style="blue"
    ))

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")

            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if not user_input.strip():
                continue

            # Show thinking indicator
            with console.status("[bold green]Thinking...[/bold green]"):
                response = asyncio.run(ai.chat(user_input))

            console.print()
            console.print(Markdown(response))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Session ended. Memories saved.[/dim]")


@cli.command()
def scout():
    """Scan for trading opportunities"""
    _, _, _, ai = get_components()

    with console.status("[bold green]Scanning markets...[/bold green]"):
        response = asyncio.run(ai.chat("@scout"))

    console.print(Markdown(response))


@cli.command()
@click.argument("symbol")
def analyze(symbol: str):
    """Deep analysis of a symbol"""
    _, _, _, ai = get_components()

    with console.status(f"[bold green]Analyzing {symbol}...[/bold green]"):
        response = asyncio.run(ai.chat(f"@analyze {symbol}"))

    console.print(Markdown(response))


@cli.command()
@click.argument("symbol")
def research(symbol: str):
    """Web research on a symbol using Claude CLI"""
    _, _, _, ai = get_components()

    with console.status(f"[bold green]Researching {symbol}...[/bold green]"):
        response = asyncio.run(ai.chat(f"@research {symbol}"))

    console.print(Markdown(response))


@cli.command()
@click.argument("symbol")
def predict(symbol: str):
    """Get ML prediction for a symbol"""
    _, _, _, ai = get_components()

    with console.status(f"[bold green]Predicting {symbol}...[/bold green]"):
        response = asyncio.run(ai.chat(f"@predict {symbol}"))

    console.print(Markdown(response))


@cli.command()
@click.argument("symbol")
def train(symbol: str):
    """Train ML model for a symbol"""
    _, _, _, ai = get_components()

    console.print(f"Training model for {symbol}...")
    console.print("This may take a few minutes...")

    response = asyncio.run(ai.chat(f"@train {symbol}"))
    console.print(Markdown(response))


@cli.command()
def risk():
    """Portfolio risk assessment"""
    _, _, _, ai = get_components()

    with console.status("[bold green]Assessing risk...[/bold green]"):
        response = asyncio.run(ai.chat("@risk"))

    console.print(Markdown(response))


@cli.command()
def regime():
    """Market regime assessment"""
    _, _, _, ai = get_components()

    with console.status("[bold green]Assessing market regime...[/bold green]"):
        response = asyncio.run(ai.chat("@regime"))

    console.print(Markdown(response))


@cli.command()
def performance():
    """Show performance metrics"""
    _, _, _, ai = get_components()

    with console.status("[bold green]Calculating performance...[/bold green]"):
        response = asyncio.run(ai.chat("@performance"))

    console.print(Markdown(response))


# ==================== Server Commands ====================

@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8420, help="Port to bind to")
def server(host: str, port: int):
    """Start the API server"""
    os.environ.setdefault("LLM_PROVIDER", "none")
    os.environ["DPOLARIS_BACKEND_HOST"] = str(host)
    os.environ["DPOLARIS_BACKEND_PORT"] = str(int(port))
    console.print(f"Starting dPolaris API server at http://{host}:{port}")
    console.print(f"LLM_PROVIDER={os.environ.get('LLM_PROVIDER', 'none')}")

    from api import run_server
    run_server(host=host, port=port)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Backend host")
@click.option("--port", default=8420, type=int, show_default=True, help="Backend port")
@click.option("--interval-health", default=60, type=int, show_default=True, help="Health check interval (seconds)")
@click.option("--interval-scan", default="30m", show_default=True, help="Scan interval (e.g. 30m, 1800, 1h)")
@click.option("--dry-run", is_flag=True, help="Do not send external notifications")
def orchestrator(host: str, port: int, interval_health: int, interval_scan: str, dry_run: bool):
    """Run the self-healing backend orchestrator."""
    from daemon.orchestrator import OrchestratorConfig, OrchestratorDaemon, parse_duration_seconds

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    scan_seconds = parse_duration_seconds(interval_scan)
    cfg = OrchestratorConfig(
        host=host,
        port=port,
        interval_health_seconds=max(5, int(interval_health)),
        interval_scan_seconds=max(60, int(scan_seconds)),
        dry_run=bool(dry_run),
    )
    daemon = OrchestratorDaemon(config=cfg)

    console.print(
        f"Starting orchestrator against http://{host}:{port} "
        f"(health={cfg.interval_health_seconds}s, scan={cfg.interval_scan_seconds}s, dry_run={cfg.dry_run})"
    )
    try:
        daemon.run_forever()
    except KeyboardInterrupt:
        daemon.stop()
        console.print("[yellow]Orchestrator stopped[/yellow]")


# ==================== Data Commands ====================

@cli.command("build-universe")
def build_universe():
    """Build Nasdaq 500 + WSB 100 + combined universe files."""
    from universe.builder import build_daily_universe_files

    with console.status("[bold green]Building daily universes...[/bold green]"):
        result = build_daily_universe_files()

    console.print("[green]Universe build completed[/green]")
    console.print(f"Nasdaq 500: {result['nasdaq500']['path']}")
    console.print(f"WSB 100: {result['wsb100']['path']}")
    console.print(f"Combined: {result['combined']['path']}")


@cli.command()
def backup():
    """Create a backup of all data"""
    config, db, memory, _ = get_components()

    backup_dir = config.data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating backup...", total=None)

        # Backup database
        db_backup = backup_dir / f"db_{timestamp}.db"
        db.backup(db_backup)
        progress.update(task, description="Database backed up...")

        # Export memory
        memory_backup = backup_dir / f"memory_{timestamp}.json"
        memory.export(memory_backup)
        progress.update(task, description="Memory exported...")

    console.print(f"[green]Backup created at {backup_dir}[/green]")


@cli.command()
def journal():
    """View trade journal"""
    _, db, _, _ = get_components()

    trades = db.get_trades(limit=20)
    stats = db.get_trade_stats()

    if not trades:
        console.print("[yellow]No trades recorded yet[/yellow]")
        return

    # Stats panel
    if stats.get("total_trades"):
        stats_text = f"""
Total Trades: {stats['total_trades']}
Win Rate: {stats.get('win_rate', 0):.1f}%
Avg Win: ${stats.get('avg_win', 0):,.2f}
Avg Loss: ${stats.get('avg_loss', 0):,.2f}
Total P/L: ${stats.get('total_pnl', 0):,.2f}
"""
        console.print(Panel(stats_text, title="Trading Statistics"))

    # Trades table
    table = Table(title="Recent Trades")
    table.add_column("Date", style="cyan")
    table.add_column("Symbol")
    table.add_column("Strategy")
    table.add_column("Direction")
    table.add_column("P/L", justify="right")

    for trade in trades:
        pnl = trade.get("pnl", 0)
        pnl_style = "green" if pnl > 0 else "red" if pnl < 0 else "white"

        table.add_row(
            trade.get("entry_date", "")[:10] if trade.get("entry_date") else "-",
            trade.get("symbol", "-"),
            trade.get("strategy", "-"),
            trade.get("direction", "-"),
            f"[{pnl_style}]${pnl:,.2f}[/{pnl_style}]" if pnl else "-",
        )

    console.print(table)


@cli.command()
def watchlist():
    """View and manage watchlist"""
    _, db, _, _ = get_components()

    items = db.get_watchlist()

    if not items:
        console.print("[yellow]Watchlist is empty[/yellow]")
        console.print("Add symbols with: dpolaris watch-add SYMBOL")
        return

    table = Table(title="Watchlist")
    table.add_column("Symbol", style="cyan")
    table.add_column("Priority")
    table.add_column("Target Entry")
    table.add_column("Thesis")

    for item in items:
        table.add_row(
            item.get("symbol", "-"),
            str(item.get("priority", 5)),
            f"${item['target_entry']:.2f}" if item.get("target_entry") else "-",
            (item.get("thesis", "")[:40] + "...") if item.get("thesis") else "-",
        )

    console.print(table)


@cli.command("watch-add")
@click.argument("symbol")
@click.option("--thesis", "-t", default="", help="Investment thesis")
@click.option("--target", "-p", type=float, help="Target entry price")
@click.option("--priority", "-r", default=5, help="Priority 1-10")
def watch_add(symbol: str, thesis: str, target: float, priority: int):
    """Add symbol to watchlist"""
    _, db, _, _ = get_components()

    db.add_to_watchlist(
        symbol=symbol.upper(),
        thesis=thesis,
        target_entry=target,
        priority=priority,
    )

    console.print(f"[green]Added {symbol.upper()} to watchlist[/green]")


@cli.command("watch-remove")
@click.argument("symbol")
def watch_remove(symbol: str):
    """Remove symbol from watchlist"""
    _, db, _, _ = get_components()

    db.remove_from_watchlist(symbol.upper())
    console.print(f"[green]Removed {symbol.upper()} from watchlist[/green]")


# ==================== Setup Command ====================

@cli.command()
def setup():
    """Initial setup wizard"""
    console.print(Panel(
        "Welcome to dPolaris AI Setup",
        title="Setup Wizard",
        border_style="blue"
    ))

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[yellow]ANTHROPIC_API_KEY not found in environment[/yellow]")
        console.print("Set it with: export ANTHROPIC_API_KEY=your_key")
    else:
        console.print("[green]ANTHROPIC_API_KEY found[/green]")

    # Create data directory
    config, _, _, _ = get_components()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Data directory: {config.data_dir}[/green]")

    # Save default config
    config_path = config.data_dir / "config" / "settings.yaml"
    if not config_path.exists():
        config.save(config_path)
        console.print(f"[green]Config saved to: {config_path}[/green]")
    else:
        console.print(f"[blue]Config exists: {config_path}[/blue]")

    # Check optional dependencies
    console.print("\n[bold]Checking optional dependencies:[/bold]")

    deps = {
        "yfinance": "Market data (Yahoo Finance)",
        "webull": "Webull broker integration",
        "pyetrade": "E*Trade broker integration",
        "ib_insync": "Interactive Brokers integration",
        "xgboost": "XGBoost ML models",
        "lightgbm": "LightGBM ML models",
    }

    for package, description in deps.items():
        try:
            __import__(package)
            console.print(f"  [green]✓[/green] {package}: {description}")
        except ImportError:
            console.print(f"  [yellow]○[/yellow] {package}: {description} (optional)")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Edit config: ~/.dpolaris_data/config/settings.yaml")
    console.print("  2. Start daemon: dpolaris start")
    console.print("  3. Chat with AI: dpolaris chat")


# ==================== Quote Command ====================

@cli.command()
@click.argument("symbol")
def quote(symbol: str):
    """Get current quote for a symbol"""
    from tools.market_data import fetch_quote

    with console.status(f"Fetching quote for {symbol}..."):
        data = asyncio.run(fetch_quote(symbol.upper()))

    if not data:
        console.print(f"[red]Could not fetch quote for {symbol}[/red]")
        return

    price = data.get("price", 0)
    prev_close = data.get("previous_close", price)
    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0

    change_style = "green" if change >= 0 else "red"

    console.print(Panel(f"""
[bold]{symbol.upper()}[/bold]

Price: ${price:,.2f}
Change: [{change_style}]${change:+,.2f} ({change_pct:+.2f}%)[/{change_style}]

Open:  ${data.get('open', 0):,.2f}
High:  ${data.get('high', 0):,.2f}
Low:   ${data.get('low', 0):,.2f}
Volume: {data.get('volume', 0):,}

52W High: ${data.get('52w_high', 0):,.2f}
52W Low:  ${data.get('52w_low', 0):,.2f}
""", title="Quote"))


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
