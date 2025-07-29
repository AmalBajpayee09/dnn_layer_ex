# phase1/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from phase1.model import Phase1Model
from phase1.utils import load_dataset, NUM_OPS

def train_model(
    trace_path,
    model_path,
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    max_len=64,
    save_path="phase1_trained.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"📁 Loading dataset...")
    X, Y = load_dataset(trace_path, model_path, max_len=max_len)
    print(f"✅ Loaded traces: {X.shape} | labels: {Y.shape}")

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("📦 Initializing model...")
    model = Phase1Model(input_dim=4, num_ops=NUM_OPS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔒 gradient clipping
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # 🔄 reduce LR
        avg_loss = total_loss / len(loader)
        print(f"📊 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    train_model(
        trace_path="data/traces/cnn_kernel_traces.json",
        model_path="generator/data/cnn_models.json"
    )
