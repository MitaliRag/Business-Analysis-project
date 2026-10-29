
import pandas as pd
from pathlib import Path

task_folder = Path(__file__).resolve().parent
project_folder = task_folder.parent
dataset_path = project_folder / "Telco_Customer_Churn_Dataset.csv"

if not dataset_path.exists():
    print("ERROR: Dataset file not found!")
    print("Please put Telco_Customer_Churn_Dataset.csv in the main project folder.")
    raise SystemExit(1)

df = pd.read_csv(dataset_path)

output = [
    "TASK 1: UNDERSTAND THE DATASET",
    "=" * 50,
    f"\nTotal Rows: {df.shape[0]}",
    f"Total Columns: {df.shape[1]}",
    "\nFirst 10 Rows:",
    df.head(10).to_string(index=False),
    "\nColumn Names and Data Types:",
    df.dtypes.to_string(),
    "\nMissing Values:",
    df.isnull().sum().to_string(),
    f"\nTotal Missing Values: {df.isnull().sum().sum()}",
    f"\nDuplicate Rows: {df.duplicated().sum()}"
]

(df.head(10)).to_csv(
    task_folder / "dataset_preview.csv", index=False
)

with open(task_folder / "task1_output.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(output))

print("\n".join(output))
print("\nTask 1 completed successfully!")
