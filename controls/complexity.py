"""
controls/complexity.py
One complexity unit for every arm, otherwise the Pareto curve is meaningless.

Convention (state this verbatim in the paper):
    literals  = number of times a feature is TESTED by the model
                  DNF    -> total literals across non-trivial clauses
                  tree   -> number of internal nodes (each node is one test)
                  linear -> number of non-zero coefficients
    concepts  = number of DISTINCT features the model touches at all
    units     = number of decision units (clauses / leaves / classes)

Trivial DNF clauses, i.e. "(True)" priors and pruned "(False)" contradictions,
are excluded from the literal count exactly as in evaluate_lucid_metrics.py.
"""
import os
import re
import sys
from typing import Dict, List

import numpy as np

_ROOT = os.environ.get("LUCID_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_FEATURE_PATTERN = re.compile(r"(?:f|z)_\d+")


def count_dnf(rules: Dict[str, List[str]]) -> Dict[str, int]:
    """Complexity of an extracted DNF rule set (output of extract_rules)."""
    units, literals, concepts = 0, 0, set()
    for _, clauses in rules.items():
        cleaned = []
        for clause in clauses:
            c = re.sub(r"\s*\[bias=[^\]]*\]", "", clause).strip()
            if c and c not in ("(no active clauses)", "(True)", "(False)"):
                cleaned.append(c)
        for clause in set(cleaned):
            units += 1
            feats = _FEATURE_PATTERN.findall(clause)
            concepts.update(feats)
            literals += len(feats)
    return {"literals": literals, "concepts": len(concepts), "units": units}


def count_tree(clf) -> Dict[str, int]:
    """Complexity of a fitted sklearn decision tree."""
    t = clf.tree_
    internal = t.children_left != -1
    return {
        "literals": int(internal.sum()),
        "concepts": int(len(np.unique(t.feature[internal]))) if internal.any() else 0,
        "units": int((~internal).sum()),
        "depth": int(clf.get_depth()),
    }


def count_linear(clf, tol: float = 1e-6) -> Dict[str, int]:
    """Complexity of a fitted (sparse) linear model."""
    coef = np.atleast_2d(clf.coef_)
    nonzero = np.abs(coef) > tol
    return {
        "literals": int(nonzero.sum()),
        "concepts": int(np.unique(np.where(nonzero)[1]).size),
        "units": int(coef.shape[0]),
    }


def count_any(model, kind: str, rules: Dict[str, List[str]] = None) -> Dict[str, int]:
    """Dispatch on arm kind: 'dnf' | 'tree' | 'linear'."""
    if kind == "dnf":
        assert rules is not None, "DNF complexity needs an extracted rule set"
        return count_dnf(rules)
    if kind == "tree":
        return count_tree(model)
    if kind == "linear":
        return count_linear(model)
    raise ValueError(f"unknown learner kind: {kind}")
