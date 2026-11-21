from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# STEP 2: DATA CLEANING
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "t20wc.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(INPUT_FILE)

print("\n========== ORIGINAL DATASET ==========")
print("Original Rows:", len(df))
print("Original Columns:", len(df.columns))

# --------------------------------------------------
# 1. REMOVE DUPLICATE RECORDS
# --------------------------------------------------

duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("\n========== DUPLICATE REMOVAL ==========")
print("Duplicates Removed:", duplicates)

# --------------------------------------------------
# 2. STANDARDIZE TEXT ATTRIBUTES
# --------------------------------------------------

text_columns = [
    "Team", "Batsman", "Wicket", "Winner", "Match"
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].str.replace(
        r"\s+", " ", regex=True
    )

print("\n========== TEXT STANDARDIZATION ==========")
print("Text attributes standardized.")

# --------------------------------------------------
# 3. SEPARATE TOTAL EXTRAS RECORDS
# --------------------------------------------------

extras_mask = df["Batsman"].eq("Total Extras")

extras_df = df.loc[extras_mask].copy()
df = df.loc[~extras_mask].copy()

extras_df.to_csv(
    OUTPUT_DIR / "t20wc_extras.csv",
    index=False
)

print("\n========== EXTRAS SEPARATION ==========")
print("Extras Records:", len(extras_df))
print("Remaining Player Records:", len(df))

# --------------------------------------------------
# 4. SEPARATE INVALID TEAM LABELS
# --------------------------------------------------

invalid_team_mask = (
    df["Team"].isna()
    | df["Team"].eq("Match Info").fillna(False)
)

invalid_team_df = df.loc[invalid_team_mask].copy()

invalid_team_df.to_csv(
    OUTPUT_DIR / "t20wc_team_review.csv",
    index=False
)

df = df.loc[~invalid_team_mask].copy()

print("\n========== INVALID TEAM HANDLING ==========")
print("Records Requiring Team Review:",
      len(invalid_team_df))
print("Remaining Records:", len(df))

# --------------------------------------------------
# 5. HANDLE MISSING / INVALID NUMERIC VALUES
# --------------------------------------------------

numeric_columns = ["R", "B", "4s", "6s", "SR"]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

print("\n========== NUMERIC CONVERSION ==========")
print(df[numeric_columns].dtypes)

print("\nMissing Numeric Values:")
print(df[numeric_columns].isna().sum())

# --------------------------------------------------
# 6. DETECT INVALID CRICKET STATISTICS
# --------------------------------------------------
# Rules:
# Runs, balls, fours, sixes cannot be negative.
# Boundary runs cannot exceed total batsman runs.
# Strike rate must be nonnegative.
# For B > 0, SR should approximately equal R/B*100.

invalid_mask = (
    (df["R"] < 0)
    | (df["B"] < 0)
    | (df["4s"] < 0)
    | (df["6s"] < 0)
    | (df["SR"] < 0)
    | ((4 * df["4s"] + 6 * df["6s"]) > df["R"])
)

calculated_sr = np.where(
    df["B"] > 0,
    df["R"] / df["B"].replace(0, np.nan) * 100,
    np.nan
)

sr_mismatch = (
    (df["B"] > 0)
    & (np.abs(df["SR"] - calculated_sr) > 0.1)
)

invalid_mask = (
    invalid_mask | sr_mismatch
).fillna(False)

invalid_stats_df = df.loc[invalid_mask].copy()

invalid_stats_df.to_csv(
    OUTPUT_DIR / "t20wc_invalid_stats.csv",
    index=False
)

print("\n========== STATISTICAL VALIDATION ==========")
print("Invalid Statistical Records:",
      len(invalid_stats_df))

df = df.loc[~invalid_mask].copy()

# --------------------------------------------------
# 7. FINAL MISSING VALUE CHECK
# --------------------------------------------------

print("\n========== FINAL MISSING VALUES ==========")
print(df.isnull().sum())

# --------------------------------------------------
# 8. FINAL DUPLICATE CHECK
# --------------------------------------------------

print("\n========== FINAL DUPLICATES ==========")
print("Duplicates:", df.duplicated().sum())

# --------------------------------------------------
# 9. SAVE CLEANED DATASET
# --------------------------------------------------

df = df.reset_index(drop=True)

output_file = OUTPUT_DIR / "t20wc_cleaned.csv"
df.to_csv(output_file, index=False)

print("\n========== CLEANED DATASET ==========")
print("Final Rows:", df.shape[0])
print("Final Columns:", df.shape[1])

print("\nCleaned Data Types:")
print(df.dtypes)

print("\nFirst 5 Cleaned Records:")
print(df.head().to_string(index=False))

print("\nFiles Saved:")
print("1.", output_file)
print("2.", OUTPUT_DIR / "t20wc_extras.csv")
print("3.", OUTPUT_DIR / "t20wc_team_review.csv")
print("4.", OUTPUT_DIR / "t20wc_invalid_stats.csv")

print("\n========== DATA CLEANING COMPLETED ==========")
