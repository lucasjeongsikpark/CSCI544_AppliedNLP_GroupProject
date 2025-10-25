
import re
from typing import Any

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _flatten(items: Any):
    if isinstance(items, (list, tuple, set)):
        for it in items:
            yield from _flatten(it)
    else:
        yield items

def accuracy(golds, preds) -> float:
    correct = 0
    for g, p in zip(golds, preds):
        if str(g) == str(p):
            correct += 1
    return correct / max(1, len(golds))

def exact_match(gold: Any, pred: Any) -> bool:
    if isinstance(gold, (list, tuple, set)):
        gset = set(normalize_text(x) for x in _flatten(gold))
        return normalize_text(pred) in gset
    return normalize_text(gold) == normalize_text(pred)

def macro_f1(golds, preds) -> float:
    # Convert labels to strings for consistency
    g2 = [str(g) for g in golds]
    p2 = [str(p) for p in preds]
    labels = sorted(set(g2) | set(p2))
    f1s = []
    for lab in labels:
        tp = sum((g==lab and p==lab) for g,p in zip(g2,p2))
        fp = sum((g!=lab and p==lab) for g,p in zip(g2,p2))
        fn = sum((g==lab and p!=lab) for g,p in zip(g2,p2))
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    return sum(f1s)/max(1,len(f1s))

def score_list_em(golds, preds) -> float:
    correct = 0
    for g,p in zip(golds, preds):
        if exact_match(g,p):
            correct += 1
    return correct / max(1, len(golds))
