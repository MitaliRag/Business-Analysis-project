
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate folders
task_folder = Path(__file__).resolve().parent
project_folder = task_folder.parent

# Use the cleaned dataset created in Task 2
dataset_path = project_folder / "Task_2_Data_Cleaning" / "cleaned_telco_churn.csv"

if not dataset_path.exists():
    print("ERROR: Cleaned dataset not found.")
    print("Complete Task 2 first and check cleaned_telco_churn.csv.")
    raise SystemExit(1)

df = pd.read_csv(dataset_path)

# Create charts folder
charts_folder = task_folder / "charts"
charts_folder.mkdir(exist_ok=True)

# Identify useful columns
tenure_col = "tenure"
charges_col = "monthlycharges"
churn_col = "churn"

# Convert numeric columns where available
for col in [tenure_col, charges_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 1. Descriptive statistics
numeric_df = df.select_dtypes(include="number")
print("\nTASK 3: EXPLORATORY DATA ANALYSIS")
print("=" * 45)

print("\nMean:")
print(numeric_df.mean())

print("\nMedian:")
print(numeric_df.median())

print("\nMode:")
print(df.mode().iloc[0])

# Save statistics
stats_path = task_folder / "eda_statistics.txt"
with open(stats_path, "w", encoding="utf-8") as file:
    file.write("MEAN\n" + numeric_df.mean().to_string())
    file.write("\n\nMEDIAN\n" + numeric_df.median().to_string())
    file.write("\n\nMODE\n" + df.mode().iloc[0].to_string())

# 2. Histograms of numeric columns
numeric_df.hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distributions of Numeric Variables")
plt.tight_layout()
plt.savefig(charts_folder / "numerical_distributions.png", dpi=150)
plt.close()

# 3. Box plot for monthly charges
if charges_col in df.columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[charges_col].dropna())
    plt.title("Monthly Charges - Box Plot")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(charts_folder / "monthly_charges_boxplot.png", dpi=150)
    plt.close()

# 4. Churn vs non-churn chart
if churn_col in df.columns:
    churn_counts = df[churn_col].astype(str).str.strip().value_counts()

    plt.figure(figsize=(7, 5))
    churn_counts.plot(kind="bar", edgecolor="black")
    plt.title("Churn vs Non-Churn Customers")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(charts_folder / "churn_distribution.png", dpi=150)
    plt.close()
else:
    print("\nChurn column not found. Check the dataset column names.")

print("\nEDA charts saved in:", charts_folder)
print("Statistics saved in:", stats_path)
print("\nTask 3 completed successfully!")

