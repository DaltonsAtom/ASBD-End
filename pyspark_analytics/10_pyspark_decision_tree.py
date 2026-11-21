from pathlib import Path
import time
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs'
INPUT = OUT / 't20wc_classification_ready.csv'
OUT.mkdir(parents=True, exist_ok=True)

spark = (SparkSession.builder.master('local[*]')
         .appName('T20WorldCupDecisionTree').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

try:
    data = spark.read.option('header', True).option('inferSchema', True).csv(str(INPUT))
    needed = ['Season', 'Innings', 'Team', 'Performance']
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}; found: {data.columns}')
    data = data.select(*[F.col(c).cast('string').alias(c) for c in needed]).dropna()
    data = data.filter(F.col('Performance').isin('Low', 'Medium', 'High'))
    count = data.count()
    if count != 2792:
        print(f'NOTE: {count} valid rows; expected 2792 from previous steps.')

    # A reproducible stratified 80/20 split.
    # Hashing the input features alone would group identical rows, so use a stable
    # per-row index after deterministic ordering; duplicate rows remain distinct.
    from pyspark.sql.window import Window
    order = Window.orderBy(*[F.col(c) for c in needed])
    indexed = data.withColumn('_row_id', F.row_number().over(order))
    indexed = indexed.withColumn('_bucket', F.pmod(F.xxhash64('_row_id', 'Performance'), F.lit(100)))
    train = indexed.filter(F.col('_bucket') < 80).drop('_row_id', '_bucket').cache()
    test = indexed.filter(F.col('_bucket') >= 80).drop('_row_id', '_bucket').cache()
    n_train, n_test = train.count(), test.count()
    if not n_train or not n_test:
        raise ValueError('Empty train or test split')

    feature_cols = ['Season', 'Innings', 'Team']
    stages = []
    for col in feature_cols:
        stages.append(StringIndexer(inputCol=col, outputCol=f'{col}_index', handleInvalid='keep'))
    stages.append(OneHotEncoder(inputCols=[f'{c}_index' for c in feature_cols],
                                outputCols=[f'{c}_vec' for c in feature_cols],
                                handleInvalid='keep'))
    stages.append(VectorAssembler(inputCols=[f'{c}_vec' for c in feature_cols], outputCol='features'))
    stages.append(StringIndexer(inputCol='Performance', outputCol='label',
                                stringOrderType='alphabetAsc', handleInvalid='keep'))
    stages.append(DecisionTreeClassifier(labelCol='label', featuresCol='features',
                                         maxDepth=5, impurity='gini', seed=42))
    pipeline = Pipeline(stages=stages)

    print('Training PySpark Decision Tree...')
    t0 = time.perf_counter()
    model = pipeline.fit(train)
    training_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    predictions = model.transform(test).cache()
    predictions.count()
    prediction_seconds = time.perf_counter() - t0

    metrics = {}
    for name, metric in [('accuracy', 'accuracy'), ('weighted_f1', 'f1'),
                         ('weighted_precision', 'weightedPrecision'),
                         ('weighted_recall', 'weightedRecall')]:
        evaluator = MulticlassClassificationEvaluator(
            labelCol='label', predictionCol='prediction', metricName=metric)
        metrics[name] = evaluator.evaluate(predictions)

    label_indexer = model.stages[-2]
    labels = label_indexer.labels
    prediction_df = (predictions.select('Performance', 'prediction')
                     .withColumn('Predicted',
                                 F.element_at(F.array(*[F.lit(x) for x in labels]),
                                              F.col('prediction').cast('int') + 1))
                     .select(F.col('Performance').alias('Actual'), 'Predicted'))
    prediction_df.toPandas().to_csv(OUT / 'pyspark_decision_tree_predictions.csv', index=False)
    confusion = (prediction_df.groupBy('Actual', 'Predicted').count().toPandas())
    confusion.to_csv(OUT / 'pyspark_decision_tree_confusion_matrix.csv', index=False)

    # Balanced accuracy: mean of per-class recalls.
    recalls = []
    for label in ['Low', 'Medium', 'High']:
        actual_total = confusion.loc[confusion['Actual'] == label, 'count'].sum()
        correct = confusion.loc[(confusion['Actual'] == label) &
                                (confusion['Predicted'] == label), 'count'].sum()
        recalls.append(float(correct / actual_total) if actual_total else 0.0)
    metrics['balanced_accuracy'] = sum(recalls) / len(recalls)

    result = {'algorithm': 'PySpark Decision Tree', 'records': count,
              'train_records': n_train, 'test_records': n_test,
              'training_seconds': training_seconds,
              'prediction_seconds': prediction_seconds, **metrics}
    pd.DataFrame([result]).to_csv(OUT / 'pyspark_decision_tree_metrics.csv', index=False)
    with open(OUT / 'pyspark_decision_tree_summary.txt', 'w', encoding='utf-8') as f:
        f.write('PYSPARK DECISION TREE CLASSIFICATION\n')
        f.write('Predictors: Season, Innings, Team; target: Performance\n')
        f.write('Max depth: 5; Gini; deterministic approximately 80/20 split\n')
        f.write('NOTE: Split and class weights differ from the previous scikit-learn run.\n')
        f.write('Results are descriptive, not a controlled platform speed comparison.\n\n')
        for k, v in result.items():
            f.write(f'{k}: {v}\n')
    print('\nRESULTS')
    for k, v in result.items():
        print(f'{k}: {v}')
    print('\nSaved 4 files in:', OUT)
finally:
    spark.stop()
