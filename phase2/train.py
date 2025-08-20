# phase2/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from phase2.model import Phase2Model
from phase2.utils import load_labels, PAD_IDX, EOS_IDX, VOCAB_SIZE
from phase1.utils import load_dataset  # For loading OPi

def train_model(
    trace_path,
    model_path,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # 🧪 Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("📁 Loading data...")
    _, X = load_dataset(trace_path, model_path, max_len=64)  # [N, 64, 10]
    Y = load_labels(model_path, max_len=50)

    # ✅ Ensure label count matches OPi count
    if len(X) != len(Y):
        min_len = min(len(X), len(Y))
        print(f"⚠️ Warning: Mismatch in data size — trimming to {min_len}")
        X = X[:min_len]
        Y = Y[:min_len]

    print(f"✅ OPi: {X.shape} | Labels: {Y.shape}")

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Phase2Model(input_dim=10, hidden_dim=256, num_layers=2, dropout=0.5).to(device)
    print(f"📦 Model initialized with input_dim=10, hidden_dim=256, num_layers=2, dropout=0.5")

    # ⚖️ Class weights to counter imbalance
    weights = torch.ones(VOCAB_SIZE)
    weights[PAD_IDX] = 0.1   # De-emphasize <PAD>
    weights[EOS_IDX] = 3.0   # Strong emphasis on <EOS>
    weights = weights.to(device)

    criterion = nn.NLLLoss(weight=weights, ignore_index=PAD_IDX, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = model(batch_x)  # [B, max_len, VOCAB_SIZE]
            preds_flat = preds.view(-1, VOCAB_SIZE)
            targets_flat = batch_y.view(-1)

            loss = criterion(preds_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🛡️ Prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📊 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "phase2_trained.pth")
            print("✅ Model improved & saved!")

        scheduler.step(avg_loss)

    print("✅ Final model saved as phase2_trained.pth")

if __name__ == "__main__":
    train_model(
        trace_path="data/traces/cnn_kernel_traces.json",
        model_path="generator/data/cnn_models.json"
    )
