# phase2/predict.py

import torch
from torch.utils.data import DataLoader, TensorDataset

from phase2.model import Phase2Model, VOCAB_SIZE
from phase2.utils import load_labels
from utils.evaluation import decode_sequence, compute_metrics
from phase1.infer import infer_opi

EOS_IDX = VOCAB_SIZE - 1  # <EOS> token index


@torch.no_grad()
def predict(
    trace_path,
    model_path,
    weights="phase2_trained.pth",
    max_len=50,
    batch_size=32
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔍 Step 1: Get OPi from Phase 1
    X = infer_opi(trace_path, model_path, weights="phase1_trained.pth", max_len=64)
    true_labels = load_labels(model_path, max_len=max_len)

    input_dim = X.shape[-1]
    print(f"📦 Phase2Model(input_dim={input_dim}, hidden_dim=256, num_layers=2)")
    model = Phase2Model(input_dim=input_dim, hidden_dim=256, num_layers=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size)

    predictions = []

    for batch in loader:
        batch_x = batch[0].to(device)            # [B, T, D]
        logits = model(batch_x)                  # [B, max_len, VOCAB_SIZE]
        pred_ids = logits.argmax(dim=-1).cpu()   # [B, max_len]

        for seq in pred_ids:
            trimmed = []
            for tok in seq:
                if tok.item() == EOS_IDX:
                    break
                trimmed.append(tok.item())
            predictions.append(torch.tensor(trimmed, dtype=torch.long))

    return predictions, true_labels


if __name__ == "__main__":
    preds, labels = predict(
        trace_path="data/traces/cnn_kernel_traces.json",
        model_path="generator/data/cnn_models.json",
        weights="phase2_trained.pth"
    )

    # 🔤 Decode for inspection
    decoded_preds = [decode_sequence(p) for p in preds]
    decoded_labels = [decode_sequence(l.view(-1)) for l in labels]

    print("\n🔍 Sample Predictions (first 5):")
    for i in range(min(5, len(decoded_preds))):
        print(f"\n◆ Sample {i+1}")
        print(f"Predicted   : {decoded_preds[i]}")
        print(f"Ground Truth: {decoded_labels[i]}")

    # 📁 Save predictions
    with open("predictions.txt", "w") as f:
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"Predicted: {pred} | True: {label}\n")
    print("\n✅ Predictions saved to predictions.txt")

    # 📊 Evaluate
    ler, f1 = compute_metrics(preds, labels)
    print(f"\n📈 Evaluation Results:\nLayer Error Rate (LER): {ler:.2f}%\nF1 Score: {f1:.2f}%")

    # 💾 Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"LER: {ler:.2f}%\nF1 Score: {f1:.2f}%\n")
    print("✅ Metrics saved to metrics.txt")
