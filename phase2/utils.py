import json
import torch
from torch.nn.utils.rnn import pad_sequence
from trace_simulator.simulate_kernel_trace import process_trace


# Define tokens
LAYER_TOKENS = [
    "conv", "relu", "batchnorm", "tanh", "sigmoid", "fc",
    "softmax", "residual", "mobilenet", "pool", "<PAD>", "<EOS>"
]

TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_IDX.items()}

PAD_IDX = TOKEN_TO_IDX["<PAD>"]
EOS_IDX = TOKEN_TO_IDX["<EOS>"]
VOCAB_SIZE = len(LAYER_TOKENS)

def encode_sequence(layer_list, max_len=50):
    """
    layer_list: list of strings, e.g. ['conv', 'relu', 'pool']
    Returns: tensor [max_len] of token IDs
    """
    ids = [TOKEN_TO_IDX.get(layer, PAD_IDX) for layer in layer_list]
    ids.append(EOS_IDX)
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode_sequence(id_tensor):
    """
    Converts list of token indices to string labels (excluding PAD and EOS).
    Accepts list[int] or tensor.
    """
    return [IDX_TO_TOKEN[int(idx)] for idx in id_tensor if idx != PAD_IDX and idx != EOS_IDX]

def load_labels(model_json_path, max_len=50):
    """
    Loads CNN model layers and converts them to token tensors.
    Returns: tensor [N, max_len]
    """
    with open(model_json_path) as f:
        models = json.load(f)

    encoded = []
    for m in models:
        layer_seq = [layer["type"].lower() for layer in m["layers"]]
        encoded.append(encode_sequence(layer_seq, max_len))

    return torch.stack(encoded)

def load_dataset(trace_path, model_path, max_len=50):
    """
    Loads kernel traces and model layers, returning padded tensors.
    Returns: (X: [N, T, 4], Y: [N, max_len])
    """
    traces = json.load(open(trace_path))
    models = json.load(open(model_path))

    X_list, Y_list = [], []

    for trace, model in zip(traces, models):
        xt = process_trace(trace)
        yt = encode_sequence([layer["type"].lower() for layer in model["layers"]], max_len)

        if xt is None or len(xt) == 0 or len(yt) == 0:
            continue

        X_list.append(xt)
        Y_list.append(yt)

    X_pad = pad_sequence(X_list, batch_first=True)
    Y_pad = pad_sequence(Y_list, batch_first=True)

    return X_pad, Y_pad
def process_trace(trace):
    """
    Converts a single trace (list of kernel events) to a tensor of shape [T, 4]
    Each row: [op_type_id, start, end, duration]
    """
    result = []
    for event in trace:
        op = TOKEN_TO_IDX.get(event["op"].lower(), PAD_IDX)
        if op == PAD_IDX: continue  # skip unknown ops
        vec = [op, event["start_time"], event["end_time"], event["duration"]]
        result.append(torch.tensor(vec, dtype=torch.float32))
    return torch.stack(result) if result else None  