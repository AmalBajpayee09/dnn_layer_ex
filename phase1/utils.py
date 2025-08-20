import json
import torch
from pathlib import Path

# 🧠 Recognized operation types (in order)
OP_TYPE_LIST = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid",
    "fc", "softmax", "residual", "mobilenet", "pool"
]

OP_TYPE_TO_ID = {op: i for i, op in enumerate(OP_TYPE_LIST)}
NUM_OPS = len(OP_TYPE_LIST)

# 🔢 Encode op type to integer ID
def encode_op(op):
    return OP_TYPE_TO_ID.get(op.lower(), -1)

# 📐 Convert trace JSON into tensor: [T, 4]
def process_trace(trace):
    result = []
    for event in trace:
        op = encode_op(event["op"])
        if op == -1:
            continue
        vec = [
            float(op),
            float(event["start_time"]),
            float(event["end_time"]),
            float(event["duration"])
        ]
        result.append(torch.tensor(vec, dtype=torch.float32))
    return torch.stack(result) if result else None

# 🎯 Convert model layers into multi-hot OPi matrix: [T, NUM_OPS]
def process_label(layers):
    result = []
    for layer in layers:
        vec = torch.zeros(NUM_OPS)
        op = encode_op(layer["type"])
        if op != -1:
            vec[op] = 1.0
        result.append(vec)
    return torch.stack(result)

# 📦 Load + process entire dataset
def load_dataset(trace_path, model_path, max_len=64):
    with open(trace_path) as f1, open(model_path) as f2:
        traces = json.load(f1)
        models = json.load(f2)

    X_list, Y_list = [], []

    for trace, model in zip(traces, models):
        xt = process_trace(trace)
        yt = process_label(model.get("layers", []))

        if xt is None or len(xt) == 0 or len(yt) == 0:
            continue

        # Truncate long sequences
        xt = xt[:max_len]
        yt = yt[:max_len]

        # Pad short sequences
        if xt.shape[0] < max_len:
            xt = torch.cat([xt, torch.zeros(max_len - xt.shape[0], xt.shape[1])], dim=0)
        if yt.shape[0] < max_len:
            yt = torch.cat([yt, torch.zeros(max_len - yt.shape[0], yt.shape[1])], dim=0)

        X_list.append(xt)
        Y_list.append(yt)

    X_pad = torch.stack(X_list)
    Y_pad = torch.stack(Y_list)

    return X_pad, Y_pad

# 💾 Save final dataset tensors to disk
def save_dataset(X, Y, output_path):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save((X, Y), output_path)
    print(f"✅ Dataset saved to {output_path}")
