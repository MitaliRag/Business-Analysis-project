
import pandas as pd
from pathlib import Path

# Find project and task folders
task_folder = Path(__file__).resolve().parent
project_folder = task_folder.parent

# Locate the dataset
dataset_path = project_folder / "Telco_Customer_Churn_Dataset.csv"

if not dataset_path.exists():
    dataset_path = task_folder / "Telco_Customer_Churn_Dataset.csv"

if not dataset_path.exists():
    print("ERROR: Dataset not found.")
    print("Place Telco_Customer_Churn_Dataset.csv in the main project folder.")
    raise SystemExit(1)

# Load dataset
df = pd.read_csv(dataset_path)

print("Original dataset shape:", df.shape)

# 1. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# 2. Remove duplicate rows
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().copy()

# 3. Handle blank values and missing values
# Convert blank or whitespace-only cells into missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

for column in df.columns:
    if df[column].isna().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_values = df[column].mode(dropna=True)

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

# 4. Save cleaned dataset
output_path = task_folder / "cleaned_telco_churn.csv"
df.to_csv(output_path, index=False)

# 5. Save cleaning report
report_path = task_folder / "task2_output.txt"

report = [
    "TASK 2: DATA CLEANING REPORT",
    "=" * 40,
    f"Original rows and columns: {pd.read_csv(dataset_path).shape}",
    f"Duplicate rows removed: {duplicates_before}",
    f"Final rows: {df.shape[0]}",
    f"Final columns: {df.shape[1]}",
    f"Missing values remaining: {df.isna().sum().sum()}",
    "\nStandardized column names:",
    ", ".join(df.columns),
]

report_text = "\n".join(report)
report_path.write_text(report_text, encoding="utf-8")

print("\n" + report_text)
print("\nTask 2 completed successfully!")
print("Cleaned dataset:", output_path)
print("Cleaning report:", report_path)
