from pathlib import Path
import pandas as pd
import time

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

# ==================================================
# STEP 6: APRIORI ASSOCIATION RULE MINING
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = OUTPUT_DIR / "t20wc_transactions.csv"

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.0
MAX_LENGTH = 3

# --------------------------------------------------
# 1. LOAD TRANSACTIONS
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

transactions = [
    [item.strip() for item in row.split(",")]
    for row in df["Items"].dropna()
]

print("\n========== TRANSACTION DATA ==========")
print("Total Transactions:", len(transactions))
print("First Transaction:", transactions[0])

# --------------------------------------------------
# 2. TRANSACTION ENCODING
# --------------------------------------------------
# Convert categorical items into Boolean columns.

encoder = TransactionEncoder()

encoded_array = encoder.fit(
    transactions
).transform(transactions)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

print("\n========== ENCODED DATA ==========")
print("Shape:", encoded_df.shape)
print("Unique Items:", len(encoded_df.columns))

# --------------------------------------------------
# 3. APRIORI FREQUENT ITEMSET GENERATION
# --------------------------------------------------

start_time = time.perf_counter()

frequent_itemsets = apriori(
    encoded_df,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_LENGTH,
    low_memory=False
)

apriori_time = time.perf_counter() - start_time

frequent_itemsets["length"] = (
    frequent_itemsets["itemsets"].apply(len)
)

frequent_itemsets = frequent_itemsets.sort_values(
    ["support", "length"],
    ascending=[False, True]
).reset_index(drop=True)

print("\n========== FREQUENT ITEMSETS ==========")
print("Total Frequent Itemsets:", len(frequent_itemsets))
print("Apriori Execution Time:",
      round(apriori_time, 6), "seconds")

itemsets_output = frequent_itemsets.copy()

itemsets_output["itemsets"] = (
    itemsets_output["itemsets"]
    .apply(lambda x: " | ".join(sorted(x)))
)

itemsets_output.to_csv(
    OUTPUT_DIR / "apriori_frequent_itemsets.csv",
    index=False
)

print("\nTop 20 Frequent Itemsets:")
print(itemsets_output.head(20).to_string(index=False))

# --------------------------------------------------
# 4. ASSOCIATION RULE GENERATION
# --------------------------------------------------

rule_start = time.perf_counter()

try:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
        num_itemsets=len(encoded_df)
    )
except TypeError:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

rule_time = time.perf_counter() - rule_start

# Keep rules with lift >= specified threshold.
rules = rules[
    rules["lift"] >= MIN_LIFT
].copy()

def attribute_names(items):
    return [item.split("=", 1)[0] for item in items]

def valid_rule(row):
    combined = row["antecedents"] | row["consequents"]
    attrs = attribute_names(combined)
    return len(attrs) == len(set(attrs))

if not rules.empty:
    rules = rules[
        rules.apply(valid_rule, axis=1)
    ].copy()

    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

print("\n========== ASSOCIATION RULES ==========")
print("Total Rules:", len(rules))
print("Rule Generation Time:",
      round(rule_time, 6), "seconds")

# --------------------------------------------------
# 5. SAVE ASSOCIATION RULES
# --------------------------------------------------

rules_output = rules.copy()

if not rules_output.empty:
    rules_output["antecedents"] = (
        rules_output["antecedents"]
        .apply(lambda x: " | ".join(sorted(x)))
    )

    rules_output["consequents"] = (
        rules_output["consequents"]
        .apply(lambda x: " | ".join(sorted(x)))
    )

    selected_columns = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction"
    ]

    rules_output = rules_output[selected_columns]

else:
    rules_output = pd.DataFrame(columns=[
        "antecedents", "consequents",
        "support", "confidence", "lift",
        "leverage", "conviction"
    ])

rules_output.to_csv(
    OUTPUT_DIR / "apriori_association_rules.csv",
    index=False
)

print("\nTop 20 Association Rules:")
print(rules_output.head(20).to_string(index=False))

# --------------------------------------------------
# 6. RULES WITH RUNS AS CONSEQUENT
# --------------------------------------------------

print("\n========== RUNS-RELATED RULES ==========")

if not rules_output.empty:
    runs_rules = rules_output[
        rules_output["consequents"].str.contains(
            "Runs=", regex=False, na=False
        )
    ]

    print(runs_rules.head(15).to_string(index=False))
else:
    print("No association rules found.")

# --------------------------------------------------
# 7. FREQUENT ITEMSET LENGTH DISTRIBUTION
# --------------------------------------------------

print("\n========== ITEMSET LENGTH DISTRIBUTION ==========")

print(
    frequent_itemsets["length"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# 8. SAVE EXECUTION SUMMARY
# --------------------------------------------------

summary = f"""
APRIORI ASSOCIATION RULE MINING SUMMARY
========================================

Total Transactions: {len(transactions)}
Unique Items: {len(encoded_df.columns)}

Minimum Support: {MIN_SUPPORT}
Minimum Confidence: {MIN_CONFIDENCE}
Minimum Lift: {MIN_LIFT}
Maximum Itemset Length: {MAX_LENGTH}

Frequent Itemsets: {len(frequent_itemsets)}
Association Rules: {len(rules)}

Apriori Execution Time: {apriori_time:.6f} seconds
Rule Generation Time: {rule_time:.6f} seconds
Total Algorithm Time: {apriori_time + rule_time:.6f} seconds
"""

with open(
    OUTPUT_DIR / "apriori_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("\n========== APRIORI COMPLETED ==========")
