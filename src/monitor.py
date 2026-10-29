
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "src" / "data" / "customer_churn.csv"
REPORT_PATH = BASE_DIR / "reports" / "drift_report.html"

# Load the customer dataset
df = pd.read_csv(DATA_PATH)

# Compare reference and current data
reference = df.iloc[:500].drop(columns=["churn"], errors="ignore")
current = df.iloc[500:].drop(columns=["churn"], errors="ignore").copy()

# Simulate a small change in customer charges
current["monthly_charges"] += 1.0

results = []

for column in reference.select_dtypes(include="number").columns:
    ref_mean = reference[column].mean()
    cur_mean = current[column].mean()
    ref_std = reference[column].std(ddof=0)

    if pd.isna(ref_std) or ref_std == 0:
        ref_std = 1.0

    shift = abs(cur_mean - ref_mean) / ref_std

    results.append({
        "Feature": column,
        "Reference Mean": round(ref_mean, 2),
        "Current Mean": round(cur_mean, 2),
        "Standardized Shift": round(shift, 3),
        "Drift Detected": "Yes" if shift >= 0.1 else "No"
    })

report = pd.DataFrame(results)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
report.to_html(REPORT_PATH, index=False)

print("Data drift monitoring completed!")
print(report.to_string(index=False))
print("\nReport saved to:", REPORT_PATH)
