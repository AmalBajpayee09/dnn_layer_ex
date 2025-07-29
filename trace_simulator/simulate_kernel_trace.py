# trace_simulator/simulate_kernel_trace.py

import json
import random
from pathlib import Path
import torch

# 📁 Input/Output Paths
CNN_MODEL_PATH = Path("generator/data/cnn_models.json")
RNN_MODEL_PATH = Path("generator/data/rnn_models.json")
CNN_TRACE_PATH = Path("data/traces/cnn_kernel_traces.json")
RNN_TRACE_PATH = Path("data/traces/rnn_kernel_traces.json")

# 🧠 Simulate kernel-level trace from model layer descriptions
def simulate_trace(layers):
    """
    Simulates start and end times for each layer's kernel execution.

    Args:
        layers (List[Dict]): List of layer dictionaries with 'type' keys.

    Returns:
        List[Dict]: Simulated trace with start_time, end_time, and duration.
    """
    trace = []
    timestamp = 0.0

    for i, layer in enumerate(layers):
        op_type = layer["type"]
        duration = round(random.uniform(0.001, 0.01), 5)

        trace.append({
            "kernel": f"{op_type}_kernel_{i}",
            "op": op_type,
            "start_time": round(timestamp, 5),
            "end_time": round(timestamp + duration, 5),
            "duration": duration
        })
        timestamp += duration

    return trace

# 📦 Generate and save all traces
def generate_traces(model_path, trace_path):
    with open(model_path) as f:
        models = json.load(f)

    all_traces = []
    for model in models:
        if "layers" not in model:
            continue
        trace = simulate_trace(model["layers"])
        all_traces.append(trace)

    with open(trace_path, "w") as f:
        json.dump(all_traces, f, indent=2)

    print(f"✅ Traces generated → {trace_path}")

# 🚀 Entry Point
def main():
    generate_traces(CNN_MODEL_PATH, CNN_TRACE_PATH)
    generate_traces(RNN_MODEL_PATH, RNN_TRACE_PATH)

if __name__ == "__main__":
    main()

# 🔧 Convert raw trace to tensor for model input
def process_trace(trace):
    """
    Converts a raw trace list into a tensor: [T, 4] — [op_idx, start, end, duration]

    Args:
        trace (List[Dict]): Kernel trace

    Returns:
        torch.Tensor: Tensor [T, 4] or None if invalid input
    """
    op_type_to_idx = {
        "conv": 0, "relu": 1, "batchnorm": 2, "tanh": 3,
        "sigmoid": 4, "fc": 5, "softmax": 6, "residual": 7,
        "mobilenet": 8, "pool": 9
    }

    if not isinstance(trace, list) or not trace:
        return None

    ops = []
    for entry in trace:
        try:
            op_idx = op_type_to_idx.get(entry["op"].lower(), 0)
            ops.append([
                float(op_idx),
                float(entry["start_time"]),
                float(entry["end_time"]),
                float(entry["duration"])
            ])
        except (KeyError, ValueError, TypeError):
            continue

    if not ops:
        return None

    return torch.tensor(ops, dtype=torch.float32)
