import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()

load_dotenv()
# accept a generic APIKEY= in .env as the OpenRouter key litellm expects
if os.environ.get("APIKEY") and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.environ["APIKEY"]


@app.command()
def run(config: Path):
    """Run the method from a YAML config; artifacts land in runs/<run_name>/."""
    from .config import RunConfig
    from .runner import run as run_method

    cfg = RunConfig.from_yaml(config)
    results = run_method(cfg)
    console.print(
        f"[bold]{cfg.run_name}[/bold] ({cfg.data.name}, {cfg.encoder.model}): "
        f"{results['pool_cv']} | cv_train {results['cv_train_accuracy']}"
    )


@app.command()
def report(runs_dir: Path = Path("runs")):
    """One table of pool_cv results across runs."""
    import json

    table = Table(title="pool_cv (test)")
    for col in ["run", "dataset", "encoder", "accuracy", "macro F1", "cv(train)"]:
        table.add_column(col)
    found = False
    for path in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        r = m.get("results", {})
        if "pool_cv" not in r:  # pre-rewrite artifacts are not comparable
            continue
        found = True
        table.add_row(
            m["run_name"],
            m["dataset"],
            m.get("encoder", "?"),
            f"{r['pool_cv']['accuracy']:.4f}",
            f"{r['pool_cv']['macro_f1']:.4f}",
            f"{r.get('cv_train_accuracy', float('nan')):.4f}",
        )
    if not found:
        console.print(f"no method runs under {runs_dir}")
        raise typer.Exit(1)
    console.print(table)


@app.command()
def diagnose(run_dir: Path):
    """Decompose a run's error: coverage vs redundancy vs fit gaps vs label noise vs artifacts."""
    from .diagnostics import diagnose_run

    d = diagnose_run(run_dir)
    console.print(f"accuracy: {d['accuracy']}  fit gaps: {d['fit_gaps']}")
    console.print(
        f"redundancy: {d['redundancy']['n_hypotheses']} hypotheses, "
        f"effective rank {d['redundancy']['effective_rank']}"
    )
    console.print("per-class F1: " + ", ".join(f"{c['class']} {c['f1']}" for c in d["per_class"]))
    for c in d["confusion_coverage"]:
        mark = "[red]⚠[/red]" if "GAP" in c["verdict"] else "·"
        console.print(f"{mark} {c['pair']}: best separator AUC {c['best_separator_auc']}")
    for f in d["length_artifact_flags"]:
        console.print(f'[red]⚠ length artifact[/red] r={f["length_corr"]}: "{f["hypothesis"][:70]}"')
    n = d["suspected_label_noise"]["confident_errors"]
    console.print(f"suspected label noise: {n} confident errors | full report: {run_dir}/diagnostics.json")


if __name__ == "__main__":
    app()
