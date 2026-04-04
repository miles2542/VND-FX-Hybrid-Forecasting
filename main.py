import os
import subprocess
import yaml
import argparse
import sys
import json

import shutil

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_python_cmd():
    """Detect if we should use 'uv run' or standard python."""
    # Priority 1: Check if 'uv' exists and if we are likely in a uv-managed project
    has_uv = shutil.which("uv") is not None
    # If running via 'uv run', UV_PYTHON or VIRTUAL_ENV is usually set
    in_uv_env = any(k in os.environ for k in ["UV_PYTHON", "VIRTUAL_ENV"])
    
    if has_uv and (in_uv_env or os.path.exists("pyproject.toml")):
        return ["uv", "run", "python"]
    return [sys.executable]

def run_step(name, command, cwd=None):
    """Orchestrate a pipeline step with env vars for R library protection."""
    print(f"\n{'='*20} STEP: {name} {'='*20}")
    
    # Ensure R scripts use our local library path
    env = os.environ.copy()
    env["R_LIBS_USER"] = "R_libs"
    
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        print(f"[SUCCESS] {name} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {name} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Hybrid FX Forecasting Pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--skip-data", action="store_true", help="Skip data loading step")
    parser.add_argument("--skip-parametric", action="store_true", help="Skip ARIMA/VAR step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostics (structural breaks) step")
    parser.add_argument("--run-hpo", action="store_true", help="Run Optuna HPO in Non-parametric stage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    py_cmd = get_python_cmd() # Detect best environment
    
    # 1. Data Loader
    if not args.skip_data:
        if not run_step("Data Loader", py_cmd + ["src/01_data_loader.py", "--config", args.config]):
            sys.exit(1)
    else:
        print("[INFO] Skipping Data Loader.")

    # 2. Diagnostics (Skeleton)
    if not args.skip_diagnostics:
        if not run_step("Diagnostics", ["Rscript", "src/02_diagnostics.R", args.config]):
            sys.exit(1)
    else:
        print("[INFO] Skipping Diagnostics.")
        
    # 3. Parametric Stage
    if not args.skip_parametric:
        if not run_step("Parametric Stage", ["Rscript", "src/03_parametric.R", args.config]):
            sys.exit(1)
    else:
        print("[INFO] Skipping Parametric Stage.")
        
    # Baselines Summary (if enabled)
    if cfg.get('baselines', {}).get('enabled', False):
        print("\n--- Baselines Summary ---")
        models = cfg['baselines'].get('models', [])
        print(f"| Active Models: {', '.join(models)}")
        print("---------------------------\n")
        
    # Read parametric summary
    active_target = cfg.get("active_target", "")
    arima_diag_path = os.path.join(cfg['paths']['results'], active_target, "arima", "diagnostics.json")
    if os.path.exists(arima_diag_path):
        with open(arima_diag_path, 'r') as f:
            arima_diag = json.load(f)
            print("\n--- ARIMA Summary (p, d, q) ---")
            for pair, res in arima_diag.items():
                print(f"| {pair:<10} | {res['order']}")
            print("-------------------------------\n")
            
    var_diag_path = os.path.join(cfg['paths']['results'], active_target, "var", "diagnostics.json")
    if os.path.exists(var_diag_path):
        with open(var_diag_path, 'r') as f:
            var_diag = json.load(f)
            print("--- VAR Summary ---")
            print(f"| Selected Lag: {var_diag['selected_lag']}")
            print("-------------------\n")

    # 4. Non-parametric Hybrid Stage (SVR/MLP)
    # Step 1: Run with defaults first to satisfy user request for verification
    cmd = py_cmd + ["src/04_nonparametric.py", "--config", args.config]
    if args.run_hpo:
        cmd.append("--run-hpo")
        
    if not run_step("Non-parametric Stage", cmd):
        sys.exit(1)
        
    # 5. Evaluation & Statistical Reporting
    if not args.skip_evaluation:
        if not run_step("Evaluation", py_cmd + ["src/05_evaluation.py", "--config", args.config]):
            sys.exit(1)
    else:
        print("[INFO] Skipping Evaluation Stage.")

    print("[SUCCESS] End-to-End Hybrid Pipeline complete (Linear + Nonlinear Correction).")

if __name__ == "__main__":
    main()
