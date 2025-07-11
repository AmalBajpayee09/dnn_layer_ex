import torch
import matplotlib.pyplot as plt
from phase1.model import Phase1Model
from phase1.utils import load_dataset, NUM_OPS

@torch.no_grad()
def visualize_attention(sample_idx=0, model_path="phase1_trained.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    trace_path = "data/traces/cnn_kernel_traces.json"
    model_path_json = "generator/data/cnn_models.json"
    X, Y = load_dataset(trace_path, model_path_json, max_len=64)
    x_sample = X[sample_idx].unsqueeze(0).to(device)

    # Load model
    model = Phase1Model(input_dim=4, num_ops=NUM_OPS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Forward pass
    embedded = model.embedding(x_sample)           # (1, T, hidden)
    lstm_out, _ = model.lstm(embedded)             # (1, T, 2*hidden)
    attn_layer = model.attn.attn                   # nn.MultiheadAttention

    attn_output, attn_weights = attn_layer(
        lstm_out, lstm_out, lstm_out,
        need_weights=True,
        average_attn_weights=False
    )

    attn_weights = attn_weights.squeeze(0).cpu()  # shape: (heads, T, T)
    print(f"Attention weights shape: {attn_weights.shape}")

    for h in range(attn_weights.shape[0]):
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_weights[h], cmap="viridis")
        plt.title(f"Head {h+1} - Self Attention")
        plt.xlabel("Key timestep")
        plt.ylabel("Query timestep")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"attention_head_{h+1}.png")
        print(f"✅ Saved attention_head_{h+1}.png")

if __name__ == "__main__":
    visualize_attention()
