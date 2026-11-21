from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# STEP 4: DATA TRANSFORMATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(
    OUTPUT_DIR / "t20wc_descriptive.csv"
)

print("\n========== ORIGINAL DATA ==========")
print("Shape:", df.shape)

numeric_cols = ["R", "B", "4s", "6s", "SR"]

# --------------------------------------------------
# 1. MIN-MAX NORMALIZATION
# --------------------------------------------------
# Formula: (X - Min) / (Max - Min)
# Converts numerical attributes into range [0, 1].

normalized_df = df.copy()

for col in numeric_cols:
    minimum = df[col].min()
    maximum = df[col].max()

    if maximum != minimum:
        normalized_df[col] = (
            (df[col] - minimum) / (maximum - minimum)
        )
    else:
        normalized_df[col] = 0.0

normalized_df.to_csv(
    OUTPUT_DIR / "t20wc_normalized.csv",
    index=False
)

print("\n========== MIN-MAX NORMALIZATION ==========")
print(normalized_df[numeric_cols].describe().loc[
    ["min", "max"]
])

# --------------------------------------------------
# 2. Z-SCORE STANDARDIZATION
# --------------------------------------------------
# Formula: Z = (X - Mean) / Standard Deviation

standardized_df = df.copy()

for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std(ddof=0)

    if std != 0:
        standardized_df[col] = (df[col] - mean) / std
    else:
        standardized_df[col] = 0.0

standardized_df.to_csv(
    OUTPUT_DIR / "t20wc_standardized.csv",
    index=False
)

print("\n========== Z-SCORE STANDARDIZATION ==========")
print(standardized_df[numeric_cols].agg(
    ["mean", "std"]
).round(4))

# --------------------------------------------------
# 3. EQUAL-WIDTH BINNING
# --------------------------------------------------
# Divide Runs into 4 equal-width intervals.

equal_width_df = df.copy()

equal_width_df["Runs_Bin"] = pd.cut(
    df["R"],
    bins=4,
    labels=["Very Low", "Low", "Medium", "High"],
    include_lowest=True
)

equal_width_df.to_csv(
    OUTPUT_DIR / "t20wc_equal_width.csv",
    index=False
)

print("\n========== EQUAL-WIDTH BINNING ==========")
print(equal_width_df["Runs_Bin"].value_counts(
    sort=False
))

# --------------------------------------------------
# 4. EQUAL-FREQUENCY BINNING
# --------------------------------------------------
# Rank values first to handle repeated run scores.
# Each bin contains approximately equal records.

equal_freq_df = df.copy()

equal_freq_df["Runs_Quantile"] = pd.qcut(
    df["R"].rank(method="first"),
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"]
)

equal_freq_df.to_csv(
    OUTPUT_DIR / "t20wc_equal_frequency.csv",
    index=False
)

print("\n========== EQUAL-FREQUENCY BINNING ==========")
print(equal_freq_df["Runs_Quantile"].value_counts(
    sort=False
))

# --------------------------------------------------
# 5. ONE-HOT ENCODING
# --------------------------------------------------
# Convert Innings into binary indicator columns.

encoded_df = pd.get_dummies(
    df,
    columns=["Innings"],
    dtype=int
)

encoded_df.to_csv(
    OUTPUT_DIR / "t20wc_encoded.csv",
    index=False
)

print("\n========== ONE-HOT ENCODING ==========")
print([
    col for col in encoded_df.columns
    if col.startswith("Innings_")
])

# --------------------------------------------------
# 6. BATTING PERFORMANCE CATEGORIES
# --------------------------------------------------
# Assumed runs-based performance categories:
# Low:    0-19 runs
# Medium: 20-49 runs
# High:   50+ runs

performance_df = df.copy()

performance_df["Performance"] = pd.cut(
    performance_df["R"],
    bins=[-np.inf, 19, 49, np.inf],
    labels=["Low", "Medium", "High"]
)

performance_df.to_csv(
    OUTPUT_DIR / "t20wc_performance.csv",
    index=False
)

print("\n========== PERFORMANCE CATEGORIES ==========")
print(performance_df["Performance"].value_counts(
    sort=False
))

# --------------------------------------------------
# 7. TRANSACTION DATA FOR APRIORI AND ACLOSE
# --------------------------------------------------
# Each transaction represents one batting innings.
# Prefixes ensure item names are unambiguous.

transactions_df = df[
    ["Season", "Innings", "R", "B", "4s", "6s", "SR"]
].copy()

transactions_df["Runs"] = pd.cut(
    df["R"],
    bins=[-np.inf, 19, 49, np.inf],
    labels=["Low", "Medium", "High"]
).astype(str)

transactions_df["Balls"] = pd.cut(
    df["B"],
    bins=[-np.inf, 10, 30, np.inf],
    labels=["Few", "Moderate", "Many"]
).astype(str)

transactions_df["Fours"] = pd.cut(
    df["4s"],
    bins=[-np.inf, 0, 3, np.inf],
    labels=["None", "Few", "Many"]
).astype(str)

transactions_df["Sixes"] = pd.cut(
    df["6s"],
    bins=[-np.inf, 0, 2, np.inf],
    labels=["None", "Few", "Many"]
).astype(str)

transactions_df["StrikeRate"] = pd.cut(
    df["SR"],
    bins=[-np.inf, 100, 150, np.inf],
    labels=["Low", "Medium", "High"]
).astype(str)

transaction_columns = [
    "Season", "Innings", "Runs",
    "Balls", "Fours", "Sixes", "StrikeRate"
]

transactions = []

for _, row in transactions_df.iterrows():
    items = [
        f"{col}={row[col]}"
        for col in transaction_columns
    ]
    transactions.append(items)

# Store one transaction per row as a string.
transaction_output = pd.DataFrame({
    "TransactionID": np.arange(1, len(transactions) + 1),
    "Items": [",".join(items) for items in transactions]
})

transaction_output.to_csv(
    OUTPUT_DIR / "t20wc_transactions.csv",
    index=False
)

print("\n========== ASSOCIATION TRANSACTIONS ==========")
print("Total Transactions:", len(transaction_output))
print(transaction_output.head().to_string(index=False))

# --------------------------------------------------
# 8. CLASSIFICATION DATA PREPARATION
# --------------------------------------------------
# Predict performance category from contextual
# features only, avoiding target leakage.
#
# R, SR, B, 4s and 6s describe the completed batting
# innings and are deliberately excluded here.

classification_df = performance_df[
    ["Season", "Innings", "Team", "Performance"]
].copy()

classification_df.to_csv(
    OUTPUT_DIR / "t20wc_classification_ready.csv",
    index=False
)

print("\n========== CLASSIFICATION PREPARATION ==========")
print("Shape:", classification_df.shape)
print("Target:", "Performance")
print("Predictors:", ["Season", "Innings", "Team"])

# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n========== TRANSFORMATION SUMMARY ==========")

output_files = [
    "t20wc_normalized.csv",
    "t20wc_standardized.csv",
    "t20wc_equal_width.csv",
    "t20wc_equal_frequency.csv",
    "t20wc_encoded.csv",
    "t20wc_performance.csv",
    "t20wc_transactions.csv",
    "t20wc_classification_ready.csv"
]

for filename in output_files:
    path = OUTPUT_DIR / filename
    result = pd.read_csv(path)
    print(
        f"{filename}: "
        f"{result.shape[0]} rows, "
        f"{result.shape[1]} columns"
    )

print("\n========== DATA TRANSFORMATION COMPLETED ==========")
