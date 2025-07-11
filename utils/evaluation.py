import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
from difflib import SequenceMatcher
from utils.constants import EOS_IDX

def compute_ler(preds, trues):
    total_err = 0
    for p, t in zip(preds, trues):
        p_trim = [x for x in p if x != EOS_IDX]
        t_trim = [x for x in t if x != EOS_IDX]
        matcher = SequenceMatcher(None, p_trim, t_trim)
        err = 1 - matcher.ratio()
        total_err += err
    return total_err / len(preds)

def evaluate(preds, trues):
    preds_flat = [x for seq in preds for x in seq if x != EOS_IDX]
    trues_flat = [x for seq in trues for x in seq if x != EOS_IDX]

    # Ensure equal length
    min_len = min(len(preds_flat), len(trues_flat))
    preds_flat = preds_flat[:min_len]
    trues_flat = trues_flat[:min_len]

    ler = compute_ler(preds, trues)
    f1 = f1_score(trues_flat, preds_flat, average="macro", zero_division=0)
    mae = mean_absolute_error(trues_flat, preds_flat)

    return {
        "LER": ler,
        "F1": f1,
        "MAE": mae
    }


if __name__ == "__main__":
    pred = [[0, 1, 9, 10], [2, 1, 9, 10]]
    true = [[0, 1, 9, 10], [2, 2, 9, 10]]
    print(evaluate(pred, true))
