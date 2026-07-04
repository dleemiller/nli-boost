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


@app.command("gepa-tune")
def gepa_tune(
    out: Path = Path("models/proposer_instruction.json"),
    tune: str = "trec:7,sst2:7",
    teacher: str = "openrouter/deepseek/deepseek-v4-pro",
    judge: str = "openrouter/deepseek/deepseek-v4-pro",
    teacher_reasoning: bool = True,
    teacher_temp: float = 1.0,
    subsamples: int = 12,
    max_calls: int = 700,
    timeout_min: float = 90.0,
    pool_size: int = 28,
    sub_size: int = 400,
):
    """Offline GEPA tuning of the GeneratePool instruction. `tune` = comma list of
    dataset:seed contexts (hold out any dataset used for the accept gate). `teacher` is the
    reflection model; --no-teacher-reasoning often diversifies proposals. Use a distinct --out
    per experiment so runs don't clobber. Stops on max_calls/timeout/`touch <out>.stop`."""
    from .gepa_tune import optimize_instruction

    specs = [(p.split(":")[0], int(p.split(":")[1])) for p in tune.split(",")]
    r = optimize_instruction(
        out,
        specs,
        reflection_model=teacher,
        judge_model=judge,
        teacher_reasoning=teacher_reasoning,
        teacher_temperature=teacher_temp,
        subsamples=subsamples,
        pool_size=pool_size,
        sub_size=sub_size,
        max_metric_calls=max_calls,
        timeout_min=timeout_min,
    )
    console.print(f"[bold]baseline reward geo-mean:[/bold] {r['baseline_geo_mean']}")
    console.print(f"[bold]tuned instruction saved:[/bold] {r['saved_to']}")
    console.print(f"[dim]stop early: touch {r['stop_file']}  |  resume: re-run this command[/dim]")
    console.print(f"\n[dim]{r['tuned_instruction']}[/dim]")


if __name__ == "__main__":
    app()
