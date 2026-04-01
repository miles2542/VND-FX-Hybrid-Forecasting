import pandas as pd
import numpy as np
import yaml
import json
import argparse
from pathlib import Path

"""
src/05_evaluation.py
--------------------
Foundation for model evaluation, metrics calculation, and reporting. 
Modular design to allow easy extension by teammate (Member D).

Metrics: MSE, MAE, RMSE, MAPE.
Output: results/evaluation/metrics_summary.csv
        results/evaluation/plot_data.json
"""

class ModelEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['paths']['results'])
        self.output_dir = self.results_dir / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_list = []
        self.plot_data = {}

    def calculate_metrics(self, actual, forecast):
        """
        Calculates standard error metrics.
        Returns a dict.
        """
        # Ensure numpy arrays
        y_true = np.array(actual)
        y_pred = np.array(forecast)
        
        # Avoid division by zero for MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
        
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        return {
            "MSE (100x Scale)": mse,
            "MAE (%)": mae,
            "RMSE (%)": rmse,
            "MAPE (%)": mape
        }

    def process_model(self, model_name):
        """
        Loads forecasts.csv for a model and computes metrics across pairs and sets.
        """
        forecast_path = self.results_dir / model_name / "forecasts.csv"
        if not forecast_path.exists():
            return
        
        df = pd.read_csv(forecast_path)
        
        for pair in df['Pair'].unique():
            pair_df = df[df['Pair'] == pair]

            # Select forecast column: prefer Hybrid (ARIMA+ML) when present
            forecast_col = 'Hybrid' if 'Hybrid' in df.columns else 'Forecast'
            
            # Store sequence data for plotting later (teammate's job)
            if pair not in self.plot_data:
                self.plot_data[pair] = {}
            
            for set_name in ['train', 'val', 'test']:
                set_df = pair_df[pair_df['Set'] == set_name]
                if set_df.empty:
                    continue
                
                metrics = self.calculate_metrics(set_df['Actual'], set_df[forecast_col])
                metrics.update({
                    "Model": model_name,
                    "Pair": pair,
                    "Set": set_name
                })
                self.metrics_list.append(metrics)
                
                # Align actual vs forecast for Jason export (Foundation for Charting)
                key = f"{model_name}_{set_name}"
                self.plot_data[pair][key] = {
                    "Date": set_df['Date'].tolist(),
                    "Actual": set_df['Actual'].tolist(),
                    "Forecast": set_df[forecast_col].tolist()
                }

    def run(self):
        """
        Iterates over all directories in results/ to discover models.
        """
        print("[INFO] Starting Evaluation Pipeline...")
        print("[NOTE] All metrics (MSE, MAE, RMSE) are computed on 100x Log-Returns (Percentage Points).")
        
        # Discover model directories
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name != "evaluation":
                print(f"  > Processing: {item.name}")
                self.process_model(item.name)
        
        # 1. Save Metrics Summary
        metrics_df = pd.DataFrame(self.metrics_list)
        metrics_csv_path = self.output_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"[SUCCESS] Metrics summary saved to: {metrics_csv_path}")
        
        # 2. Save Plot Data (Foundation for Member D)
        plot_json_path = self.output_dir / "plot_data.json"
        with open(plot_json_path, 'w') as f:
            json.dump(self.plot_data, f)
        print(f"[SUCCESS] Prepared plot data sequences in: {plot_json_path}")
        
        # 3. Print a quick highlight
        test_metrics = metrics_df[metrics_df['Set'] == 'test'].sort_values(['Pair', 'MSE (100x Scale)'])
        print("\n" + "="*70)
        print("TEST SET PERFORMANCE HIGHLIGHTS (Units: % Log-Returns)")
        print("="*70)
        print(test_metrics[['Pair', 'Model', 'MSE (100x Scale)', 'MAE (%)']].to_string(index=False))
        print("="*70 + "\n")

# =============================================================================
# DIEBOLD-MARIANO TEST STUB
# =============================================================================
def diebold_mariano_test(actual, forecast_a, forecast_b, h=1, criterion='mse'):
    """
    STUB: Diebold-Mariano test for predictive accuracy comparison.
    Member D: Implement the actual loss-differencing logic and S-statistic here.
    
    Citation: Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    """
    # Boilerplate check: ensure same length
    assert len(actual) == len(forecast_a) == len(forecast_b), "Sequence lengths must match"
    
    # h = forecast horizon (1 for this project)
    # criterion = loss function (mse or mae)
    
    print(f"[STUB] Diebold-Mariano comparison initiated (h={h}, loss='{criterion}')")
    
    # TODO: Member D to implement:
    # 1. Calculate loss differential d_t = L(e_at) - L(e_bt)
    # 2. Estimate long-run variance of d_t (HAC estimator / Newey-West)
    # 3. Compute DM statistic (S)
    # 4. Return p-value
    
    return {
        "statistic": 0.0,
        "p_value": 1.0,
        "significance": "Stub Only - Implementation Pending"
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting performance.")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    evaluator.run()

if __name__ == "__main__":
    main()
