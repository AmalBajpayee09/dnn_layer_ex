import torch
from sklearn.metrics import f1_score

# 🔡 Token Definitions
LAYER_TOKENS = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", "fc",
    "softmax", "residual", "mobilenet", "pool", "<PAD>", "<EOS>"
]

TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

PAD_IDX = TOKEN_TO_IDX["<PAD>"]
EOS_IDX = TOKEN_TO_IDX["<EOS>"]
VOCAB_SIZE = len(LAYER_TOKENS)

# 🔓 Decode a tensor of indices
def decode_sequence(id_tensor):
    result = []
    for idx in id_tensor:
        idx = int(idx)
        if idx == EOS_IDX:
            break
        if idx != PAD_IDX:
            result.append(IDX_TO_TOKEN.get(idx, "<UNK>"))
    return result

# 🧾 Edit Distance (Levenshtein)
def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # deletion
                    dp[i][j - 1],    # insertion
                    dp[i - 1][j - 1] # substitution
                )
    return dp[m][n]

# 📊 Evaluate Layer Error Rate (LER) and F1 Score
def compute_metrics(predictions, targets):
    total_ler = 0.0
    total_f1 = []
    count = 0

    for pred_seq, true_seq in zip(predictions, targets):
        pred = decode_sequence(pred_seq)
        true = decode_sequence(true_seq)

        if not true:
            continue

        ler = edit_distance(pred, true) / max(1, len(true))
        total_ler += ler

        # F1 (micro-averaged on truncated match length)
        match_len = min(len(pred), len(true))
        if match_len == 0:
            continue

        y_true = [TOKEN_TO_IDX.get(tok, PAD_IDX) for tok in true[:match_len]]
        y_pred = [TOKEN_TO_IDX.get(tok, PAD_IDX) for tok in pred[:match_len]]
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        total_f1.append(f1)
        count += 1

    avg_ler = (total_ler / count) * 100 if count else 100.0
    avg_f1 = (sum(total_f1) / count) * 100 if count else 0.0
    return avg_ler, avg_f1
