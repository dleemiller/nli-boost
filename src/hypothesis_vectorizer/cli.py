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
def compare(run_a: Path, run_b: Path):
    """Paired McNemar test: is run_b's accuracy difference from run_a real or noise?"""
    from .compare import compare_runs

    r = compare_runs(run_a, run_b)
    mc = r["mcnemar"]
    console.print(
        f"[bold]{r['run_a']}[/bold] {r['acc_a']:.4f} {r['ci_a']}  vs  "
        f"[bold]{r['run_b']}[/bold] {r['acc_b']:.4f} {r['ci_b']}   "
        f"({r['dataset']} seed {r['seed']}, n={r['n_test']})"
    )
    console.print(
        f"delta {r['delta_b_minus_a']:+.4f}  |  discordant {mc['discordant']} "
        f"(A-only {mc['b_only_A']}, B-only {mc['c_only_B']})  |  McNemar p={mc['p_value']:.4f}"
    )
    verdict = (
        "[green]significant[/green] (p<0.05)"
        if r["significant_at_05"]
        else "[yellow]not significant[/yellow] — within noise"
    )
    console.print(f"verdict: {verdict}")


if __name__ == "__main__":
    app()
