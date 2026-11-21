from pathlib import Path
from itertools import combinations
from collections import defaultdict
import time
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = OUT / 't20wc_transactions.csv'
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.0

start = time.perf_counter()
raw = pd.read_csv(INPUT)
transactions = [frozenset(str(s).split(',')) for s in raw['Items'].dropna()]
transactions = [frozenset(i.strip() for i in t if i.strip()) for t in transactions]
N = len(transactions)
if not N:
    raise ValueError('No transactions found')
min_count = __import__('math').ceil(MIN_SUPPORT * N)


# Vertical TID representation: each item maps to transaction indices.
vertical = defaultdict(set)
for tid, transaction in enumerate(transactions):
    for item in transaction:
        vertical[item].add(tid)


# Apriori: generate all frequent itemsets, with no max_len restriction.
frequent = {}  # frozenset(items) -> frozenset(transaction IDs)
level = {}
for item, tids in vertical.items():
    if len(tids) >= min_count:
        level[frozenset([item])] = frozenset(tids)
frequent.update(level)
levels = {1: len(level)}
k = 2
while level:
    previous = set(level)
    ordered = sorted(previous, key=lambda s: tuple(sorted(s)))
    candidates = set()
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            candidate = a | b
            if len(candidate) != k:
                continue
            if all(frozenset(sub) in previous for sub in combinations(candidate, k - 1)):
                candidates.add(candidate)
    next_level = {}
    for candidate in candidates:
        parts = list(candidate)
        tids = set(vertical[parts[0]])
        for item in parts[1:]:
            tids.intersection_update(vertical[item])
            if len(tids) < min_count:
                break
        if len(tids) >= min_count:
            next_level[candidate] = frozenset(tids)
    if next_level:
        levels[k] = len(next_level)
    frequent.update(next_level)
    level = next_level
    k += 1


# Closure of X is the intersection of every transaction containing X.
closure_cache = {}
for itemset, tids in frequent.items():
    common = set(transactions[next(iter(tids))])
    for tid in tids:
        common.intersection_update(transactions[tid])
    closure_cache[itemset] = frozenset(common)


# Deduplicate closed itemsets by their closure.
closed = {}
for itemset, closure in closure_cache.items():
    closed[closure] = frequent[itemset]

generators = {}
for itemset, tids in frequent.items():
    if len(itemset) == 1:
        minimal = len(tids) < N
    else:
        minimal = all(
            len(frequent[frozenset(sub)]) != len(tids)
            for sub in combinations(itemset, len(itemset) - 1)
        )
    if minimal:
        generators[itemset] = closure_cache[itemset]


# Exact closed implication basis: generator -> closure(generator) - generator.
# Confidence is 1 by the definition of closure.
rules = []
for generator, closure in generators.items():
    consequent = closure - generator
    if not consequent:
        continue
    support_count = len(frequent[generator])
    consequent_tids = set(range(N))
    for item in consequent:
        consequent_tids.intersection_update(vertical[item])
    support = support_count / N
    consequent_support = len(consequent_tids) / N
    confidence = 1.0
    lift = confidence / consequent_support
    if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
        rules.append({
            'antecedents': ' | '.join(sorted(generator)),
            'consequents': ' | '.join(sorted(consequent)),
            'support': support,
            'confidence': confidence,
            'lift': lift,
            'generator_length': len(generator),
            'closure_length': len(closure)
        })

closed_df = pd.DataFrame([
    {'itemsets': ' | '.join(sorted(items)), 'support': len(tids)/N,
     'support_count': len(tids), 'length': len(items)}
    for items, tids in closed.items()
]).sort_values(['support', 'length'], ascending=[False, True])
closed_df.to_csv(OUT / 'aclose_closed_itemsets.csv', index=False)

generator_df = pd.DataFrame([
    {'generator': ' | '.join(sorted(items)),
     'closure': ' | '.join(sorted(closure)),
     'support': len(frequent[items])/N}
    for items, closure in generators.items()
]).sort_values('support', ascending=False)
generator_df.to_csv(OUT / 'aclose_minimal_generators.csv', index=False)

rule_columns = ['antecedents', 'consequents', 'support', 'confidence',
                'lift', 'generator_length', 'closure_length']
rules_df = pd.DataFrame(rules, columns=rule_columns)
if not rules_df.empty:
    rules_df = rules_df.sort_values(['lift', 'support'], ascending=False)
rules_df.to_csv(OUT / 'aclose_closed_rules.csv', index=False)

elapsed = time.perf_counter() - start
summary = (
    f'ACLOSE-STYLE CLOSED FREQUENT ITEMSET MINING\n'
    f'Transactions: {N}\nDistinct items: {len(vertical)}\n'
    f'Minimum support: {MIN_SUPPORT} (count >= {min_count})\n'
    f'Minimum confidence: {MIN_CONFIDENCE}\nMinimum lift: {MIN_LIFT}\n'
    f'All frequent itemsets: {len(frequent)}\n'
    f'Closed frequent itemsets: {len(closed)}\n'
    f'Minimal generators: {len(generators)}\n'
    f'Exact closed implication rules: {len(rules_df)}\n'
    f'Frequent itemsets by size: {levels}\n'
    f'Elapsed time: {elapsed:.6f} seconds\n'
    f'NOTE: Exact closure rules have confidence 1.0; this is not the '
    f'complete set of approximate rules with confidence >= {MIN_CONFIDENCE}.\n'
)
(OUT / 'aclose_summary.txt').write_text(summary, encoding='utf-8')
print(summary)
print('TOP 15 CLOSED ITEMSETS')
print(closed_df.head(15).to_string(index=False))
print('\nTOP 15 EXACT CLOSED RULES')
print(rules_df.head(15).to_string(index=False))
print('\nACLOSE COMPLETED')
