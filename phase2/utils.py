# phase2/utils.py

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from trace_simulator.simulate_kernel_trace import process_trace  # assumed implemented elsewhere

# 🔡 Vocabulary Tokens
LAYER_TOKENS = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", "fc",
    "softmax", "residual", "mobilenet", "pool", "<PAD>", "<EOS>"
]

TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

PAD_IDX = TOKEN_TO_IDX["<PAD>"]
EOS_IDX = TOKEN_TO_IDX["<EOS>"]
VOCAB_SIZE = len(LAYER_TOKENS)


# 🔒 Encode token sequence to index tensor
def encode_sequence(layer_list, max_len=50):
    ids = [TOKEN_TO_IDX.get(layer, PAD_IDX) for layer in layer_list]
    ids.append(EOS_IDX)  # Append <EOS>
    ids = ids[:max_len] + [PAD_IDX] * max(0, max_len - len(ids))  # Pad if needed
    return torch.tensor(ids, dtype=torch.long)


# 🔓 Decode tensor to list of tokens
def decode_sequence(id_tensor):
    result = []
    for idx in id_tensor:
        idx = int(idx)
        if idx < 0 or idx >= VOCAB_SIZE:
            continue
        tok = IDX_TO_TOKEN[idx]
        if tok == "<EOS>":
            break
        if tok != "<PAD>":
            result.append(tok)
    return result


# 📁 Load model labels from json
def load_labels(model_json_path, max_len=50):
    with open(model_json_path) as f:
        models = json.load(f)

    encoded = []
    for model in models:
        if "layers" not in model:
            continue
        layer_seq = [layer["type"].lower() for layer in model["layers"] if "type" in layer]
        encoded_seq = encode_sequence(layer_seq, max_len)
        encoded.append(encoded_seq)

    return torch.stack(encoded)


# 🧪 Load processed traces and labels
def load_dataset(trace_path, model_path, max_len=50):
    with open(trace_path) as f:
        traces = json.load(f)
    with open(model_path) as f:
        models = json.load(f)

    X_list, Y_list = [], []

    for trace, model in zip(traces, models):
        xt = process_trace(trace)
        if "layers" not in model:
            continue
        layer_seq = [layer["type"].lower() for layer in model["layers"] if "type" in layer]
        yt = encode_sequence(layer_seq, max_len)

        if xt is None or len(xt) == 0 or len(yt) == 0:
            continue

        X_list.append(xt)
        Y_list.append(yt)

    X_pad = pad_sequence(X_list, batch_first=True)
    Y_pad = pad_sequence(Y_list, batch_first=True)

    return X_pad, Y_pad
