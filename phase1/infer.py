# phase1/infer.py

import torch
from phase1.model import Phase1Model
from phase1.utils import NUM_OPS, load_dataset

@torch.no_grad()
def infer_opi(trace_path, model_path, weights="phase1_trained.pth", max_len=64, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X, _ = load_dataset(trace_path, model_path, max_len=max_len)
    model = Phase1Model(input_dim=4, num_ops=NUM_OPS).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    outputs = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size].to(device)
        out = model(batch).cpu()
        outputs.append(out)

    opi = torch.cat(outputs, dim=0)
    return opi
