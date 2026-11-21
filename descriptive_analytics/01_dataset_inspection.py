from pathlib import Path
import pandas as pd

# ==================================================
# STEP 1: DATASET INSPECTION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "t20wc.csv"
df = pd.read_csv(DATASET_PATH)

# 1. Dataset dimensions
print("\n========== DATASET DIMENSIONS ==========")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# 2. First and last five records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

# 3. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. Dataset information and data types
print("\n========== DATASET INFORMATION ==========")
df.info()

# 5. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Total Duplicates:", df.duplicated().sum())

# 7. Basic statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").to_string())

# 8. Unique values in each attribute
print("\n========== UNIQUE VALUES ==========")
print(df.nunique())

# 9. Check invalid numeric entries
print("\n========== INVALID NUMERIC VALUES ==========")

numeric_columns = ["R", "B", "4s", "6s", "SR"]

for col in numeric_columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    invalid = df.loc[converted.isna(), col]

    print(f"\nColumn: {col}")
    print("Invalid Count:", len(invalid))
    print("Invalid Values:")
    print(invalid.value_counts())

# 10. Season-wise record distribution
print("\n========== SEASON DISTRIBUTION ==========")
print(df["Season"].value_counts().sort_index())

# 11. Team-wise record distribution
print("\n========== TEAM DISTRIBUTION ==========")
print(df["Team"].value_counts())

print("\n========== INSPECTION COMPLETED ==========")
