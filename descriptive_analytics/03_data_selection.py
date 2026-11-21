from pathlib import Path
import pandas as pd

# ==================================================
# STEP 3: DATA SELECTION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "t20wc_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
df = pd.read_csv(INPUT_FILE)

print("\n========== ORIGINAL CLEANED DATA ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# --------------------------------------------------
# 1. DESCRIPTIVE ANALYTICS DATASET
# --------------------------------------------------
# Select attributes relevant to batting performance,
# team comparisons and year-wise statistics.

descriptive_columns = [
    "Season", "Matchid", "Innings", "Team",
    "Batsman", "R", "B", "4s", "6s", "SR"
]

descriptive_df = df[descriptive_columns].copy()

descriptive_df.to_csv(
    OUTPUT_DIR / "t20wc_descriptive.csv",
    index=False
)

print("\n========== DESCRIPTIVE DATA ==========")
print("Shape:", descriptive_df.shape)
print(descriptive_df.head().to_string(index=False))

# --------------------------------------------------
# 2. VISUALIZATION DATASET
# --------------------------------------------------

visualization_columns = [
    "Season", "Matchid", "Innings", "Team",
    "Batsman", "Wicket", "R", "B",
    "4s", "6s", "SR", "Winner"
]

visualization_df = df[visualization_columns].copy()

visualization_df.to_csv(
    OUTPUT_DIR / "t20wc_visualization.csv",
    index=False
)

print("\n========== VISUALIZATION DATA ==========")
print("Shape:", visualization_df.shape)

# --------------------------------------------------
# 3. ASSOCIATION RULE MINING DATASET
# --------------------------------------------------
# Each row represents one batsman's innings.
# Continuous numerical values will be discretized
# into categories during data transformation.

association_columns = [
    "Season", "Matchid", "Innings",
    "Team", "R", "B", "4s", "6s", "SR"
]

association_df = df[association_columns].copy()

association_df.to_csv(
    OUTPUT_DIR / "t20wc_association.csv",
    index=False
)

print("\n========== ASSOCIATION DATA ==========")
print("Shape:", association_df.shape)

# --------------------------------------------------
# 4. CLASSIFICATION DATASET
# --------------------------------------------------
# Potential target: batting performance category.
# Runs and strike rate are candidate features for
# descriptive analysis, but target-derived features
# must be excluded when training a classifier.

classification_columns = [
    "Season", "Innings", "Team",
    "R", "B", "4s", "6s", "SR"
]

classification_df = df[classification_columns].copy()

classification_df.to_csv(
    OUTPUT_DIR / "t20wc_classification.csv",
    index=False
)

print("\n========== CLASSIFICATION DATA ==========")
print("Shape:", classification_df.shape)

# --------------------------------------------------
# 5. ATTRIBUTE SELECTION SUMMARY
# --------------------------------------------------

print("\n========== ATTRIBUTE SELECTION SUMMARY ==========")

datasets = {
    "Descriptive": descriptive_df,
    "Visualization": visualization_df,
    "Association": association_df,
    "Classification": classification_df
}

for name, dataset in datasets.items():
    print(f"\n{name} Dataset")
    print("Rows:", dataset.shape[0])
    print("Columns:", dataset.shape[1])
    print("Attributes:", dataset.columns.tolist())

# --------------------------------------------------
# 6. VERIFY DATA INTEGRITY
# --------------------------------------------------

print("\n========== DATA INTEGRITY CHECK ==========")

for name, dataset in datasets.items():
    print(
        f"{name}: "
        f"Rows={len(dataset)}, "
        f"Missing={dataset.isnull().sum().sum()}, "
        f"Duplicates={dataset.duplicated().sum()}"
    )

print("\n========== DATA SELECTION COMPLETED ==========")
