# phase2/predict.py

import torch
from torch.utils.data import DataLoader, TensorDataset

from phase2.model import Phase2Model
from phase2.utils import decode_sequence, load_labels, VOCAB_SIZE
from phase1.infer import infer_opi

@torch.no_grad()
def predict(trace_path, model_path, weights="phase2_trained.pth", max_len=50, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer OPi
    X = infer_opi(trace_path, model_path, weights="phase1_trained.pth", max_len=64)
    true_labels = load_labels(model_path, max_len=max_len)

    input_dim = X.shape[-1]
    print(f"📦 Initializing Phase 2 model with input_dim={input_dim} and hidden_dim=128")
    model = Phase2Model(input_dim=input_dim, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size)

    preds_all = []

    for batch in loader:
        batch_x = batch[0].to(device)  # [B, T, D]
        batch_preds = model(batch_x).argmax(dim=-1).cpu().tolist()
        preds_all.extend(batch_preds)

    return preds_all, true_labels
if __name__ == "__main__":
    preds, labels = predict(
        trace_path="data/traces/cnn_kernel_traces.json",
        model_path="generator/data/cnn_models.json"
    )

    # Decode predictions
    decoded_preds = [decode_sequence(p) for p in preds]
    decoded_labels = [decode_sequence(l) for l in labels]

    # Print some results
    for i in range(5):
        print(f"Predicted: {decoded_preds[i]} | True: {decoded_labels[i]}")

    # Save predictions to file
    with open("predictions.txt", "w") as f:
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"Predicted: {pred} | True: {label}\n")
    print("✅ Predictions saved to predictions.txt")

    # Example evaluation (you can implement your own metrics)
    correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_preds)
    print(f"Accuracy: {accuracy:.4f}")

    # Save accuracy to file
    with open("accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    print("✅ Accuracy saved to accuracy.txt")  