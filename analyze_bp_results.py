import json
import glob
import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

# Mapping series keys to readable names
TICKER_NAMES = {
    "USDVND_RET": "USD/VND",
    "EURVND_RET": "EUR/VND",
    "JPYVND_RET": "JPY/VND",
    "CNYVND_RET": "CNY/VND"
}

console = Console()

def load_all_results():
    pattern = os.path.join("results", "diagnostics", "report_h*.json")
    files = glob.glob(pattern)
    all_data = []
    
    for f in files:
        with open(f, 'r') as jf:
            data = json.load(jf)
            meta = data.get("_meta", {})
            h = meta.get("bp_h")
            bp_type = meta.get("bp_type")
            
            for key, val in data.items():
                if key == "_meta": continue
                
                sb = val.get("structural_breaks", {})
                duration = meta.get("durations", {}).get(key, 0)
                
                all_data.append({
                    "h": h,
                    "type": bp_type,
                    "series": TICKER_NAMES.get(key, key),
                    "n_breaks": sb.get("n_breaks", 0),
                    "found": sb.get("found", False),
                    "dates": sb.get("dates", []),
                    "bic": sb.get("bic", [0])[0] if isinstance(sb.get("bic"), list) else sb.get("bic", 0),
                    "duration": duration,
                    "filename": os.path.basename(f)
                })
    return pd.DataFrame(all_data)

def display_summary(df):
    console.print(Panel("[bold cyan]Bai-Perron Structural Break Diagnostic Summary[/bold cyan]", expand=False))
    
    # Execution Time Summary
    time_table = Table(title="Performance Metrics by Configuration", box=None)
    time_table.add_column("Config (h, type)", style="cyan")
    time_table.add_column("Avg Duration (s)", justify="right")
    time_table.add_column("Total Config Time (s)", justify="right")
    
    perf = df.groupby(['h', 'type'])['duration'].agg(['mean', 'sum']).reset_index()
    for _, row in perf.sort_values(['type', 'h'], ascending=[True, False]).iterrows():
        time_table.add_row(f"h={row['h']}, {row['type']}", f"{row['mean']:.2f}", f"{row['sum']:.2f}")
    
    console.print(time_table)
    console.print("-" * 50)

    # Breaks Discovery Table
    breaks_table = Table(title="Structural Breaks Discovery Matrix")
    breaks_table.add_column("Series", style="bold white")
    
    # Create columns for each (h, type) combo
    combos = df[['h', 'type']].drop_duplicates().sort_values(['type', 'h'], ascending=[True, False])
    for _, row in combos.iterrows():
        col_name = f"h={row['h']}\n({row['type']})"
        breaks_table.add_column(col_name, justify="center")
    
    series_list = df['series'].unique()
    for s in series_list:
        row_data = [s]
        for _, c_row in combos.iterrows():
            match = df[(df['series'] == s) & (df['h'] == c_row['h']) & (df['type'] == c_row['type'])]
            if not match.empty:
                n = int(match.iloc[0]['n_breaks'])
                style = "bold green" if n > 0 else "dim white"
                row_data.append(Text(str(n), style=style))
            else:
                row_data.append("-")
        breaks_table.add_row(*row_data)
    
    console.print(breaks_table)

    # Detailed Breakthroughs (if any)
    significant = df[df['n_breaks'] > 0]
    if not significant.empty:
        console.print("\n[bold yellow]Detected Structural Break Registry:[/bold yellow]")
        for _, row in significant.iterrows():
            console.print(f" • [cyan]{row['series']}[/cyan] ({row['type']}, h={row['h']}): {', '.join(row['dates'])}")
    else:
        console.print("\n[bold red]Note:[/bold red] No structural breaks were detected across all 6 configurations using the BIC selection criterion.")
        console.print("[dim]This suggests a high degree of regime stability in the log-returns (mean) and squared returns (variance) for these VND-centric pairs, or that the BIC penalty favored the zero-break model for the given sample size.[/dim]")

def main():
    try:
        df = load_all_results()
        if df.empty:
            console.print("[bold red]No result files found in results/diagnostics/[/bold red]")
            return
        
        display_summary(df)
        
    except Exception as e:
        console.print(f"[bold red]Error analyzing results: {str(e)}[/bold red]")

if __name__ == "__main__":
    main()
