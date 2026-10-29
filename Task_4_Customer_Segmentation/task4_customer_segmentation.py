
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
required_columns = ["tenure", "monthlycharges"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("ERROR: Required columns not found:", missing_columns)
    print("Available columns:", df.columns.tolist())
    raise SystemExit(1)

# Convert columns to numeric
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["monthlycharges"] = pd.to_numeric(
    df["monthlycharges"], errors="coerce"
)

df = df.dropna(subset=["tenure", "monthlycharges"]).copy()

# Create customer tenure groups
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, float("inf")],
    labels=["0-12 Months", "13-36 Months", "37+ Months"]
)

# Create charts folder
charts_folder = task_folder / "charts"
charts_folder.mkdir(parents=True, exist_ok=True)

# 1. Pie chart showing customer distribution
group_counts = (
    df["tenure_group"]
    .value_counts()
    .reindex(["0-12 Months", "13-36 Months", "37+ Months"])
)

plt.figure(figsize=(8, 6))
plt.pie(
    group_counts,
    labels=group_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Customer Distribution by Tenure Group")
plt.tight_layout()
plt.savefig(
    charts_folder / "tenure_distribution.png",
    dpi=150
)
plt.close()

# 2. Clustered-style bar chart comparing average monthly charges
avg_charges = (
    df.groupby("tenure_group", observed=False)["monthlycharges"]
    .mean()
    .reindex(["0-12 Months", "13-36 Months", "37+ Months"])
)

plt.figure(figsize=(8, 6))
avg_charges.plot(kind="bar", edgecolor="black")
plt.title("Average Monthly Charges by Tenure Group")
plt.xlabel("Customer Tenure Group")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    charts_folder / "monthly_charges_by_tenure.png",
    dpi=150
)
plt.close()

# Save the segmentation summary
summary = pd.DataFrame({
    "customer_count": group_counts,
    "average_monthly_charges": avg_charges
})

summary_path = task_folder / "customer_segmentation_summary.csv"
summary.to_csv(summary_path)

print("\nCUSTOMER SEGMENTATION SUMMARY")
print("=" * 40)
print(summary)
print("\nCharts saved in:", charts_folder)
print("Summary saved in:", summary_path)
print("\nTask 4 completed successfully!")

