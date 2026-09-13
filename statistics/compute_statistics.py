#!/usr/bin/env python3
"""Recompute the confidence intervals and exact tests reported in the article.

    python statistics/compute_statistics.py
    python statistics/compute_statistics.py path/to/counts.json

The counts in counts.json are those reported in the article: the recall of the
temporal-graph method and of the direct model on the controlled injections, per course
and defect type, with the paired outcome of the two methods on the same injections;
and the flagged and confirmed issues per course on naturally occurring issues.

Only the Python standard library is used.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054          # standard normal quantile for a two-sided 95 % interval
METHODS = ("temporal_graph", "direct_model")
LABEL = {"temporal_graph": "temporal-graph method", "direct_model": "direct model"}


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test: a binomial test of b discordant pairs out of
    b + c against probability 1/2."""
    n = b + c
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for the 2x2 table [[a, b], [c, d]]: the total
    probability of the tables with the same margins that are no more likely than the
    observed one."""
    row1, row2, col1, n = a + b, c + d, a + c, a + b + c + d

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = probability(a)
    tables = range(max(0, col1 - row2), min(row1, col1) + 1)
    return min(1.0, sum(p for p in map(probability, tables) if p <= observed * (1 + 1e-9)))


def describe(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f}, 95% Wilson CI {lo:.3f}-{hi:.3f}"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("counts.json")
    counts = json.loads(path.read_text(encoding="utf-8"))
    recall, precision = counts["recall"], counts["precision"]
    per_cell = recall["injections_per_course_and_type"]

    print("Recall on the controlled injections")
    totals = {}
    for method in METHODS:
        cells = [hits for course in recall[method].values() for hits in course.values()]
        totals[method] = (sum(cells), per_cell * len(cells))
        print(f"  {LABEL[method]:<22} {describe(*totals[method])}")

    paired = recall["paired"]
    both, graph_only = paired["both_identified"], paired["temporal_graph_only"]
    direct_only, neither = paired["direct_model_only"], paired["both_missed"]
    n_paired = both + graph_only + direct_only + neither
    if totals["temporal_graph"] != (both + graph_only, n_paired):
        raise SystemExit("the paired counts do not add up to the recall of the temporal-graph method")
    if totals["direct_model"] != (both + direct_only, n_paired):
        raise SystemExit("the paired counts do not add up to the recall of the direct model")
    print(f"  paired on {n_paired} injections: both identified {both}, both missed {neither}, "
          f"temporal-graph method only {graph_only}, direct model only {direct_only}")
    print(f"  exact McNemar test on {graph_only + direct_only} discordant pairs: "
          f"p = {mcnemar_exact(graph_only, direct_only):.5f}")

    print("\nPrecision on naturally occurring issues")
    pooled = {}
    for method in METHODS:
        rates = []
        for course, c in precision[method].items():
            rates.append(c["confirmed"] / c["flagged"])
            print(f"  {LABEL[method]:<22} {course:<21} {describe(c['confirmed'], c['flagged'])}")
        confirmed = sum(c["confirmed"] for c in precision[method].values())
        flagged = sum(c["flagged"] for c in precision[method].values())
        pooled[method] = (confirmed, flagged)
        print(f"  {LABEL[method]:<22} macro-average {sum(rates) / len(rates):.3f}, "
              f"micro-average {confirmed}/{flagged} = {confirmed / flagged:.3f}")
    (k1, n1), (k2, n2) = pooled["temporal_graph"], pooled["direct_model"]
    print(f"  exact Fisher test on the pooled counts: p = {fisher_exact(k1, n1 - k1, k2, n2 - k2):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
