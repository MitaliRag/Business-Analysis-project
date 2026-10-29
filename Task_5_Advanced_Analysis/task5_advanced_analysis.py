
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate folders
task_folder = Path(__file__).resolve().parent
project_folder = task_folder.parent

# Load cleaned dataset from Task 2
dataset_path = (
    project_folder
    / "Task_2_Data_Cleaning"
    / "cleaned_telco_churn.csv"
)

if not dataset_path.exists():
    print("ERROR: Cleaned dataset not found.")
    print("Please complete Task 2 first.")
    raise SystemExit(1)

df = pd.read_csv(dataset_path)

# Check required columns
required_columns = [
    "tenure", "monthlycharges", "churn",
    "gender", "seniorcitizen",
    "paymentmethod", "contract"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("ERROR: Required columns not found:", missing_columns)
    print("Available columns:", df.columns.tolist())
    raise SystemExit(1)

# Convert numeric columns
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["monthlycharges"] = pd.to_numeric(
    df["monthlycharges"], errors="coerce"
)

# Create charts folder
charts_folder = task_folder / "charts"
charts_folder.mkdir(parents=True, exist_ok=True)

# Normalize churn labels for grouping
df["churn"] = df["churn"].astype(str).str.strip()

# 1. Churn by tenure group
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, float("inf")],
    labels=["0-12 Months", "13-36 Months", "37+ Months"]
)

tenure_churn = pd.crosstab(
    df["tenure_group"], df["churn"], normalize="index"
) * 100

tenure_churn.plot(kind="bar", figsize=(9, 6))
plt.title("Churn Percentage by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(charts_folder / "churn_by_tenure.png", dpi=150)
plt.close()

# 2. Churn by gender
gender_churn = pd.crosstab(
    df["gender"], df["churn"], normalize="index"
) * 100

gender_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Percentage by Gender")
plt.xlabel("Gender")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(charts_folder / "churn_by_gender.png", dpi=150)
plt.close()

# 3. Churn by senior citizen status
senior_churn = pd.crosstab(
    df["seniorcitizen"], df["churn"], normalize="index"
) * 100

senior_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Percentage by Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(charts_folder / "churn_by_senior_citizen.png", dpi=150)
plt.close()

# 4. Churn by payment method
payment_churn = pd.crosstab(
    df["paymentmethod"], df["churn"], normalize="index"
) * 100

payment_churn.plot(kind="bar", figsize=(10, 6))
plt.title("Churn Percentage by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(charts_folder / "churn_by_payment_method.png", dpi=150)
plt.close()

# 5. Churn by contract type
contract_churn = pd.crosstab(
    df["contract"], df["churn"], normalize="index"
) * 100

contract_churn.plot(kind="bar", figsize=(8, 5))
plt.title("Churn Percentage by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(charts_folder / "churn_by_contract.png", dpi=150)
plt.close()

# 6. Average monthly charges by churn status
charges_churn = df.groupby("churn")["monthlycharges"].mean()

plt.figure(figsize=(8, 5))
charges_churn.plot(kind="bar", edgecolor="black")
plt.title("Average Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(charts_folder / "monthly_charges_by_churn.png", dpi=150)
plt.close()

# Save analysis tables
tables = {
    "churn_by_tenure.csv": tenure_churn,
    "churn_by_gender.csv": gender_churn,
    "churn_by_senior_citizen.csv": senior_churn,
    "churn_by_payment_method.csv": payment_churn,
    "churn_by_contract.csv": contract_churn,
    "average_monthly_charges_by_churn.csv": charges_churn.to_frame()
}

for filename, table in tables.items():
    table.to_csv(task_folder / filename)

# Create text report
report_lines = [
    "TASK 5: ADVANCED CUSTOMER CHURN ANALYSIS",
    "=" * 50,
    "\nChurn by Tenure Group (percentage):",
    tenure_churn.round(2).to_string(),
    "\n\nChurn by Gender (percentage):",
    gender_churn.round(2).to_string(),
    "\n\nChurn by Senior Citizen Status (percentage):",
    senior_churn.round(2).to_string(),
    "\n\nChurn by Payment Method (percentage):",
    payment_churn.round(2).to_string(),
    "\n\nChurn by Contract Type (percentage):",
    contract_churn.round(2).to_string(),
    "\n\nAverage Monthly Charges by Churn:",
    charges_churn.round(2).to_string()
]

report_path = task_folder / "task5_advanced_analysis_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

print("\n".join(report_lines))
print("\nCharts saved in:", charts_folder)
print("Report saved in:", report_path)
print("\nTask 5 completed successfully!")
