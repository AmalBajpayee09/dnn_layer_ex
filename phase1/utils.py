# phase1/utils.py

import json
import torch
import numpy as np
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

OP_TYPE_LIST = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", 
    "fc", "softmax", "residual", "mobilenet", "pool"
]

OP_TYPE_TO_ID = {op: i for i, op in enumerate(OP_TYPE_LIST)}
NUM_OPS = len(OP_TYPE_LIST)

def encode_op(op):
    """Map op string to numeric id."""
    return OP_TYPE_TO_ID.get(op.lower(), -1)

def process_trace(trace):
    """
    Converts a single trace (list of kernel events) to a tensor of shape [T, 4]
    Each row: [op_type_id, start, end, duration]
    """
    result = []
    for event in trace:
        op = encode_op(event["op"])
        if op == -1: continue  # skip unknown ops
        vec = [op, event["start_time"], event["end_time"], event["duration"]]
        result.append(torch.tensor(vec, dtype=torch.float32))
    return torch.stack(result) if result else None

def process_label(layers):
    """
    Converts DNN layers to a [T, NUM_OPS] multi-hot label matrix OPi
    """
    result = []
    for layer in layers:
        vec = torch.zeros(NUM_OPS)
        op = encode_op(layer["type"])
        if op != -1:
            vec[op] = 1.0
        result.append(vec)
    return torch.stack(result)
def load_dataset(trace_path, model_path, max_len=64):
    traces = json.load(open(trace_path))
    models = json.load(open(model_path))

    X_list, Y_list = [], []

    for trace, model in zip(traces, models):
        xt = process_trace(trace)
        yt = process_label(model["layers"])
        if xt is None or len(xt) == 0 or len(yt) == 0:
            continue

        # truncate longer
        xt = xt[:max_len]
        yt = yt[:max_len]

        # pad shorter
        if xt.shape[0] < max_len:
            pad_len = max_len - xt.shape[0]
            xt = torch.cat([xt, torch.zeros(pad_len, xt.shape[1])], dim=0)

        if yt.shape[0] < max_len:
            pad_len = max_len - yt.shape[0]
            yt = torch.cat([yt, torch.zeros(pad_len, yt.shape[1])], dim=0)

        X_list.append(xt)
        Y_list.append(yt)

    X_pad = torch.stack(X_list)
    Y_pad = torch.stack(Y_list)

    return X_pad, Y_pad
def save_dataset(X, Y, output_path):
    """
    Save the dataset tensors to a file.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save((X, Y), output_path)
    print(f"Dataset saved to {output_path}")
