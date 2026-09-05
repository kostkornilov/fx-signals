from pathlib import Path
from typing import Annotated

import typer

from fx_signal.baseline import run_baseline
from fx_signal.data import fetch_snapshot, load_yaml
from fx_signal.indicator_search import run_indicator_report, run_indicator_search
from fx_signal.public_context import fetch_public_context
from fx_signal.research import run_research
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


if __name__ == "__main__":
    app()
