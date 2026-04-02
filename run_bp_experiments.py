import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# --- Configuration ---
CONFIG_PATH = "configs/pipeline_config.yaml"
R_SCRIPT = "src/02_diagnostics.R"
MAX_CONCURRENT_EXPERIMENTS = 3  # 3 experiments * 4 R-workers = 12 total threads (Safe for 15 limit)
R_INTERNAL_WORKERS = 4          # Number of series processed in parallel within R

EXPERIMENTS = [
    {"h": 0.15, "type": "mean",     "skip_stats": 1, "out": "report_h0.15_mean.json"},
    {"h": 0.10, "type": "mean",     "skip_stats": 1, "out": "report_h0.10_mean.json"},
    {"h": 0.05, "type": "mean",     "skip_stats": 1, "out": "report_h0.05_mean.json"},
    {"h": 0.15, "type": "variance", "skip_stats": 1, "out": "report_h0.15_variance.json"},
    {"h": 0.10, "type": "variance", "skip_stats": 1, "out": "report_h0.10_variance.json"},
    {"h": 0.05, "type": "variance", "skip_stats": 1, "out": "report_h0.05_variance.json"},
]

console = Console()

def run_experiment(exp):
    """Executes a single R diagnostic run with specified parameters."""
    h_val = exp["h"]
    type_val = exp["type"]
    skip_stats = exp["skip_stats"]
    out_file = exp["out"]
    
    cmd = [
        "uv", "run", "Rscript", R_SCRIPT,
        CONFIG_PATH,
        str(h_val),
        type_val,
        str(skip_stats),
        out_file,
        str(R_INTERNAL_WORKERS)
    ]
    
    start_time = time.time()
    try:
        # We use check=True to raise an error if R fails
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        return {
            "config": f"h={h_val}, {type_val}",
            "status": "[bold green]Success[/bold green]",
            "duration": f"{duration:.2f}s",
            "output_file": out_file,
            "error": None
        }
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        return {
            "config": f"h={h_val}, {type_val}",
            "status": "[bold red]Failed[/bold red]",
            "duration": f"{duration:.2f}s",
            "output_file": out_file,
            "error": e.stderr
        }

def main():
    console.print("[bold blue]Bai-Perron Structural Break Orchestrator[/bold blue]")
    console.print(f"Concurrency: {MAX_CONCURRENT_EXPERIMENTS} parallel runs, each using {R_INTERNAL_WORKERS} threads.")
    console.print(f"Total target threads: {MAX_CONCURRENT_EXPERIMENTS * R_INTERNAL_WORKERS} / 15 allowed.")
    console.print("-" * 50)

    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Experiments", total=len(EXPERIMENTS))
        
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EXPERIMENTS) as executor:
            future_to_exp = {executor.submit(run_experiment, exp): exp for exp in EXPERIMENTS}
            
            for future in as_completed(future_to_exp):
                res = future.result()
                results.append(res)
                progress.advance(main_task)
                
                # Dynamic update on arrival
                status_color = "green" if "Success" in res["status"] else "red"
                console.print(f"[{status_color}]FINISH[/{status_color}] {res['config']} in {res['duration']}")

    # Final Summary Table
    table = Table(title="Bai-Perron Experiment Results")
    table.add_column("Configuration", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Result File", style="dim")

    for res in sorted(results, key=lambda x: x["config"]):
        table.add_row(res["config"], res["status"], res["duration"], res["output_file"])

    console.print(table)
    
    # Check for errors and display if any
    errors = [r for r in results if r["error"]]
    if errors:
        console.print("\n[bold red]Error Details:[/bold red]")
        for err in errors:
            console.print(f"[bold]{err['config']}:[/bold]\n{err['error']}")

if __name__ == "__main__":
    main()
