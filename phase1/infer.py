# phase1/infer.py

import torch
from phase1.model import Phase1Model
from phase1.utils import NUM_OPS, load_dataset

@torch.no_grad()
def infer_opi(
    trace_path: str,
    model_path: str,
    weights: str = "phase1_trained.pth",
    max_len: int = 64,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Phase 1 inference: Converts kernel traces into OPi embeddings using the trained Phase1Model.

    Args:
        trace_path (str): Path to trace JSON file.
        model_path (str): Path to model JSON file.
        weights (str): Path to Phase1 trained model weights.
        max_len (int): Max sequence length (padding/truncation).
        batch_size (int): Batch size for inference.

    Returns:
        torch.Tensor: Tensor of shape [N, T, D] representing OPi sequences.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 📥 Load trace inputs [N, T, 4]
    X, _ = load_dataset(trace_path, model_path, max_len=max_len)

    # 🧠 Load trained model
    model = Phase1Model(input_dim=4, num_ops=NUM_OPS).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    outputs = []

    # 🔄 Inference loop
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size].to(device)
        out = model(batch).cpu()  # [B, T, D]
        outputs.append(out)

    # 🔗 Concatenate all batches
    opi = torch.cat(outputs, dim=0)  # [N, T, D]
    return opi
