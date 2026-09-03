from pathlib import Path
from typing import Annotated

import typer

from fx_signal.baseline import run_baseline
from fx_signal.data import fetch_snapshot

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


if __name__ == "__main__":
    app()
