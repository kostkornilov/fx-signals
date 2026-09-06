from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from fx_signal.backtest_report import run_backtest_report
from fx_signal.baseline import run_baseline
from fx_signal.data import fetch_snapshot, load_yaml
from fx_signal.external import from_public_context
from fx_signal.indicator_search import run_indicator_report, run_indicator_search
from fx_signal.public_context import fetch_public_context
from fx_signal.research import run_research
from fx_signal.signal_snapshot import run_signal_snapshot
from fx_signal.train import run_summary, run_train

app = typer.Typer(help="Explainable FX signal experiments")
data_app = typer.Typer(help="Download and prepare market data")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def fetch(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/data.yaml"),
    force: Annotated[bool, typer.Option(help="Replace an existing local snapshot")] = False,
) -> None:
    path = fetch_snapshot(config, force=force)
    typer.echo(f"Snapshot ready: {path}")


@data_app.command("fetch-context")
def fetch_context(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/indicator_search.yaml"),
    force: Annotated[bool, typer.Option(help="Replace an existing context snapshot")] = False,
) -> None:
    loaded = load_yaml(config)
    path = fetch_public_context(loaded, config.resolve().parent.parent, force=force)
    typer.echo(f"Public context snapshot ready: {path}")


@data_app.command("fetch-external")
def fetch_external(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/research_external.yaml"),
    force: Annotated[bool, typer.Option(help="Replace existing public and external snapshots")] = False,
) -> None:
    loaded = load_yaml(config)
    repo_root = config.resolve().parent.parent
    public_path = fetch_public_context(loaded, repo_root, force=force)
    external_path = Path(loaded.get("external_path", "data/raw/external/market_series.csv"))
    if not external_path.is_absolute():
        external_path = repo_root / external_path
    if external_path.exists() and not force:
        typer.echo(f"External snapshot already present: {external_path}")
        return
    context = pd.read_csv(public_path, parse_dates=["effective_date", "available_at"])
    series = from_public_context(context, source=str(loaded.get("external_source", "moex")))
    external_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(external_path, index=False)
    typer.echo(f"External snapshot ready: {external_path}")


@app.command("baseline")
def baseline(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/baseline.yaml"),
) -> None:
    path = run_baseline(config)
    typer.echo(f"Baseline metrics written to: {path}")


@app.command("train")
def train(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
    exp_name: Annotated[str, typer.Option(help="Experiment id written to the journal")] = "logreg_ab",
    method: Annotated[str | None, typer.Option(help="Override model/method")] = None,
) -> None:
    path = run_train(config, exp_name=exp_name, method=method)
    typer.echo(f"Experiment journal written to: {path}")


@app.command("summary")
def summary(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
) -> None:
    path = run_summary(config)
    typer.echo(f"Summary table written to: {path}")


@app.command("backtest-report")
def backtest_report(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/backtest_report.yaml"),
    first_test_year: Annotated[int | None, typer.Option()] = None,
    oot_start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Recompute the strict walk-forward backtest and publish its reports."""
    path = run_backtest_report(
        config, first_test_year=first_test_year, oot_start=oot_start, end=end
    )
    typer.echo(f"Backtest report written to: {path}")


@app.command("indicator-search")
def indicator_search(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/indicator_search.yaml"),
) -> None:
    path = run_indicator_search(config)
    typer.echo(f"Indicator search report written to: {path}")


@app.command("indicator-report")
def indicator_report(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/indicator_search.yaml"),
) -> None:
    path = run_indicator_report(config)
    typer.echo(f"Indicator search report refreshed: {path}")


@app.command("research")
def research(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/research.yaml"),
) -> None:
    path = run_research(config)
    typer.echo(f"Research report written to: {path}")


@app.command("signals")
def signals(
    as_of: Annotated[str, typer.Option(help="Point-in-time cutoff in YYYY-MM-DD format")],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/signals.yaml"),
    output: Annotated[Path | None, typer.Option(help="Destination CSV path")] = None,
) -> None:
    """Calculate the final communication signals as they looked at one cutoff."""
    path = run_signal_snapshot(config, as_of=as_of, output_path=output)
    count = len(pd.read_csv(path))
    typer.echo(f"Signal snapshot written to: {path} ({count} signal(s))")


if __name__ == "__main__":
    app()
