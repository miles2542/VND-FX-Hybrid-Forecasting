import os
import json
import pandas as pd
import yaml
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

def load_config(path="configs/pipeline_config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export summary to HTML and SVG")
    args = parser.parse_args()

    # Use record=True if export is requested
    console = Console(record=args.export)
    
    cfg = load_config()
    target = cfg.get("active_target", "UNKNOWN")
    results_root = os.path.join(cfg.get("paths", {}).get("results", "results"), target)
    
    # 1. Header
    title_text = Text.assemble(
        ("\n", ""),
        ("  ", "blue"),
        (f"HYBRID FORECASTING ANALYSIS: ", "bold white"),
        (f"{target}", "bold cyan italic"),
        ("\n", "")
    )
    console.print(title_text)

    # --- 2. Diagnostics ---
    diag_path = os.path.join(results_root, "diagnostics", "initial_report.json")
    if os.path.exists(diag_path):
        console.print("[bold underline]1. Time Series Diagnostics[/bold underline]", style="cyan")
        diag_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE, padding=(0, 2))
        diag_table.add_column("Pair", style="dim")
        diag_table.add_column("ADF", justify="center")
        diag_table.add_column("KPSS", justify="center")
        diag_table.add_column("Breaks (Mean)", justify="center")
        diag_table.add_column("Breaks (Var)", justify="center")

        with open(diag_path, "r") as f:
            diag_data = json.load(f)

        for key, val in diag_data.items():
            if key.startswith("_"): continue
            adf_status = "[green]Stat[/green]" if "Stationary" in str(val.get("adf", {}).get("decision", "N/A")) else "[red]Non-Stat[/red]"
            kpss_status = "[green]Stat[/green]" if "Stationary" in str(val.get("kpss", {}).get("decision", "N/A")) else "[red]Non-Stat[/red]"
            b_mean = val.get("structural_breaks", {}).get("mean", {}).get("n_breaks", 0)
            b_var = val.get("structural_breaks", {}).get("variance", {}).get("n_breaks", 0)
            diag_table.add_row(key, adf_status, kpss_status, f"[yellow]{b_mean}[/yellow]" if b_mean > 0 else "0", f"[yellow]{b_var}[/yellow]" if b_var > 0 else "0")
        
        console.print(diag_table)

    # --- 3. Performance Tables ---
    metrics_path = os.path.join(results_root, "evaluation", "metrics_summary.csv")
    if os.path.exists(metrics_path):
        console.print("\n[bold underline]2. Test Set Performance Comparison[/bold underline]", style="green")
        df = pd.read_csv(metrics_path)
        test_df = df[df["Set"] == "test"].copy()
        
        for pair in test_df["Pair"].unique():
            pair_data = test_df[test_df["Pair"] == pair].sort_values("MSE (100x Scale)")
            best_mse = pair_data.iloc[0]["MSE (100x Scale)"]
            rw_row = pair_data[pair_data["Model"] == "baseline_rw"]
            rw_mse = rw_row.iloc[0]["MSE (100x Scale)"] if not rw_row.empty else None

            p_table = Table(title=f"[italic white]Pair: {pair}[/italic white]", show_header=True, header_style="bold green", box=box.SIMPLE_HEAD)
            p_table.add_column("Rank", justify="right", style="dim")
            p_table.add_column("Model Name", width=25)
            p_table.add_column("MSE (100x)", justify="right")
            p_table.add_column("MAE (%)", justify="right")
            p_table.add_column("vs. RW (%)", justify="right")

            for i, (_, row) in enumerate(pair_data.iterrows()):
                name, mse, mae = row["Model"].upper().replace("_", " "), row["MSE (100x Scale)"], row["MAE (%)"]
                perf_vs_rw = f"[{( 'green' if (d := ((mse - rw_mse)/rw_mse)*100) < 0 else 'red' )}]{d:+.2f}%[/{( 'green' if d < 0 else 'red' )}]" if rw_mse and name != "BASELINE RW" else "-"
                style = "bold on #263238" if mse == best_mse else ("italic on #333300" if "BASELINE RW" in name else None)
                p_table.add_row(str(i+1), ("⭐ " if mse == best_mse else "") + name, f"{mse:.6f}", f"{mae:.4f}", perf_vs_rw, style=style)
            console.print(p_table)
            console.print("")

    # --- 4. Export logic ---
    if args.export:
        export_dir = os.path.join(results_root, "evaluation", "reports")
        os.makedirs(export_dir, exist_ok=True)
        
        html_file = os.path.join(export_dir, f"report_{target}.html")
        svg_file = os.path.join(export_dir, f"report_{target}.svg")
        
        console.save_html(html_file)
        console.save_svg(svg_file)
        
        console.print(f"\n[bold green]SUCCESS:[/bold green] Exported colored reports to:")
        console.print(f" > HTML: {html_file}")
        console.print(f" > SVG:  {svg_file}")

if __name__ == "__main__":
    main()
