# trace_simulator/simulate_kernel_trace.py

import json
import random
from pathlib import Path

CNN_MODEL_PATH = Path("generator/data/cnn_models.json")
RNN_MODEL_PATH = Path("generator/data/rnn_models.json")

CNN_TRACE_PATH = Path("data/traces/cnn_kernel_traces.json")
RNN_TRACE_PATH = Path("data/traces/rnn_kernel_traces.json")


def simulate_trace(layers):
    trace = []
    timestamp = 0.0

    for i, layer in enumerate(layers):
        op_type = layer["type"]
        duration = round(random.uniform(0.001, 0.01), 5)  # simulate kernel execution time

        trace.append({
            "kernel": f"{op_type}_kernel_{i}",
            "op": op_type,
            "start_time": round(timestamp, 5),
            "end_time": round(timestamp + duration, 5),
            "duration": duration
        })
        timestamp += duration
    return trace


def generate_traces(model_path, trace_path):
    with open(model_path) as f:
        models = json.load(f)

    all_traces = []
    for model in models:
        layers = model["layers"]
        trace = simulate_trace(layers)
        all_traces.append(trace)

    with open(trace_path, "w") as f:
        json.dump(all_traces, f, indent=2)

    print(f"✅ Traces generated → {trace_path}")


def main():
    generate_traces(CNN_MODEL_PATH, CNN_TRACE_PATH)
    generate_traces(RNN_MODEL_PATH, RNN_TRACE_PATH)


if __name__ == "__main__":
    main()


# 🔧 Added for Phase 2 inference integration
def process_trace(trace):
    """
    Converts a single raw trace (list of dicts) into a tensor of [T, 4] → [start_time, end_time, duration, op_index]
    """
    import torch

    op_type_to_idx = {
        "conv": 0, "relu": 1, "batchnorm": 2, "tanh": 3,
        "sigmoid": 4, "fc": 5, "softmax": 6, "residual": 7,
        "mobilenet": 8, "pool": 9
    }

    if not trace or not isinstance(trace, list):
        return None

    ops = []
    for entry in trace:
        try:
            op_idx = op_type_to_idx.get(entry["op"].lower(), 0)
            ops.append([
                float(entry["start_time"]),
                float(entry["end_time"]),
                float(entry["duration"]),
                float(op_idx)
            ])
        except Exception:
            continue

    if not ops:
        return None

    return torch.tensor(ops, dtype=torch.float32)
