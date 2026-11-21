from pathlib import Path
import time
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

# ==================================================
# STEP 9A: PYSPARK ASSOCIATION RULE MINING
# LOCAL APRIORI VS SPARK FP-GROWTH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_FILE = OUTPUT_DIR / "t20wc_transactions.csv"

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MAX_LENGTH = 3
REPEATS = 5

# --------------------------------------------------
# 1. LOAD TRANSACTIONS
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

transactions = [
    [item.strip() for item in row.split(",")]
    for row in df["Items"]
]

print("\n========== DATASET ==========")
print("Transactions:", len(transactions))

# --------------------------------------------------
# 2. LOCAL APRIORI BENCHMARK
# --------------------------------------------------

encoder = TransactionEncoder()

encoded = encoder.fit(
    transactions
).transform(transactions)

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.columns_
)

local_times = []
local_itemsets = None
local_rules = None

for i in range(REPEATS):
    start = time.perf_counter()

    itemsets = apriori(
        encoded_df,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_LENGTH
    )

    try:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=MIN_CONFIDENCE,
            num_itemsets=len(encoded_df)
        )
    except TypeError:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=MIN_CONFIDENCE
        )

    elapsed = time.perf_counter() - start
    local_times.append(elapsed)

    local_itemsets = itemsets
    local_rules = rules

print("\n========== LOCAL APRIORI ==========")
print("Frequent Itemsets:", len(local_itemsets))
print("Raw Association Rules:", len(local_rules))
print("Average Time:", sum(local_times) / REPEATS)

# --------------------------------------------------
# 3. INITIALIZE PYSPARK
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("T20WorldCupAssociationMining")
    .master("local[*]")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Create Spark transaction dataset.
spark_data = [
    (int(i), items)
    for i, items in enumerate(transactions, start=1)
]

spark_df = spark.createDataFrame(
    spark_data,
    ["TransactionID", "items"]
)

spark_df = spark_df.cache()
spark_df.count()

print("\n========== PYSPARK DATA ==========")
print("Spark Transactions:", spark_df.count())

# --------------------------------------------------
# 4. PYSPARK FP-GROWTH BENCHMARK
# --------------------------------------------------
# Spark FP-Growth mines all frequent itemset lengths.
# Filter to length <= 3 for a comparable itemset count.

spark_times = []
spark_itemsets = None
spark_rules = None

for i in range(REPEATS):
    start = time.perf_counter()

    fp = FPGrowth(
        itemsCol="items",
        minSupport=MIN_SUPPORT,
        minConfidence=MIN_CONFIDENCE,
        numPartitions=2
    )

    model = fp.fit(spark_df)

    itemsets_result = model.freqItemsets.cache()
    rules_result = model.associationRules.cache()

    itemsets_result.count()
    rules_result.count()

    elapsed = time.perf_counter() - start
    spark_times.append(elapsed)

    if i == REPEATS - 1:
        spark_itemsets = itemsets_result.toPandas()
        spark_rules = rules_result.toPandas()

    itemsets_result.unpersist()
    rules_result.unpersist()

print("\n========== PYSPARK FP-GROWTH ==========")

spark_itemsets["length"] = (
    spark_itemsets["items"].apply(len)
)

comparable_itemsets = spark_itemsets[
    spark_itemsets["length"] <= MAX_LENGTH
].copy()

print("All Frequent Itemsets:", len(spark_itemsets))
print("Itemsets Length <= 3:", len(comparable_itemsets))
print("Raw Association Rules:", len(spark_rules))
print("Average Time:", sum(spark_times) / REPEATS)

# --------------------------------------------------
# 5. VERIFY FREQUENT ITEMSETS
# --------------------------------------------------
# Apriori and FP-Growth should discover the same
# frequent itemsets at equal support thresholds.

local_set = {
    frozenset(items)
    for items in local_itemsets["itemsets"]
}

spark_set = {
    frozenset(items)
    for items in comparable_itemsets["items"]
}

same_itemsets = local_set == spark_set

print("\n========== RESULT VALIDATION ==========")
print("Frequent Itemsets Match:", same_itemsets)
print("Only in Apriori:", len(local_set - spark_set))
print("Only in Spark:", len(spark_set - local_set))

# --------------------------------------------------
# 6. PERFORMANCE COMPARISON
# --------------------------------------------------

local_avg = sum(local_times) / REPEATS
spark_avg = sum(spark_times) / REPEATS

speedup = local_avg / spark_avg

print("\n========== PERFORMANCE COMPARISON ==========")
print("Local Apriori Average:", round(local_avg, 6))
print("Spark FP-Growth Average:", round(spark_avg, 6))
print("Spark Speedup:", round(speedup, 4), "x")

if speedup > 1:
    print("Spark was faster in this benchmark.")
else:
    print("Local Apriori was faster in this benchmark.")

# --------------------------------------------------
# 7. SAVE COMPARISON
# --------------------------------------------------

comparison_df = pd.DataFrame({
    "Algorithm": [
        "Local Apriori",
        "PySpark FP-Growth"
    ],
    "Average_Time_Seconds": [
        local_avg,
        spark_avg
    ],
    "Minimum_Support": [
        MIN_SUPPORT,
        MIN_SUPPORT
    ],
    "Minimum_Confidence": [
        MIN_CONFIDENCE,
        MIN_CONFIDENCE
    ],
    "Transactions": [
        len(transactions),
        len(transactions)
    ]
})

comparison_df.to_csv(
    OUTPUT_DIR / "pyspark_association_comparison.csv",
    index=False
)

spark_itemsets.to_csv(
    OUTPUT_DIR / "pyspark_frequent_itemsets.csv",
    index=False
)

spark_rules.to_csv(
    OUTPUT_DIR / "pyspark_association_rules.csv",
    index=False
)

with open(
    OUTPUT_DIR / "pyspark_association_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "PYSPARK ASSOCIATION MINING SUMMARY\n"
        "==================================\n"
        f"Transactions: {len(transactions)}\n"
        f"Support: {MIN_SUPPORT}\n"
        f"Confidence: {MIN_CONFIDENCE}\n"
        f"Apriori Itemsets (max length 3): {len(local_set)}\n"
        f"Spark Itemsets (max length 3): {len(spark_set)}\n"
        f"Itemsets Match: {same_itemsets}\n"
        f"Local Average Time: {local_avg:.6f}s\n"
        f"Spark Average Time: {spark_avg:.6f}s\n"
        f"Spark Speedup: {speedup:.4f}x\n"
    )

spark_df.unpersist()
spark.stop()

print("\n========== PYSPARK ASSOCIATION COMPLETED ==========")
