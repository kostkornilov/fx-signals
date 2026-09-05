from pathlib import Path
from typing import Annotated

import typer

from fx_signal.baseline import run_baseline
from fx_signal.data import fetch_snapshot
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


if __name__ == "__main__":
    app()
