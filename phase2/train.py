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
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("📁 Loading data...")
    _, X = load_dataset(trace_path, model_path, max_len=64)  # [N, 64, 10] ← OPi from Phase 1
    Y = load_labels(model_path, max_len=50)                  # [N, 50] ← label token sequence
    print(f"✅ OPi: {X.shape} | Labels: {Y.shape}")

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Phase2Model(input_dim=10, hidden_dim=128).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)  # [B, max_len, VOCAB_SIZE]

            # reshape for loss: (B*T, V) vs (B*T)
            preds_flat = preds.view(-1, VOCAB_SIZE)
            targets_flat = batch_y.view(-1)

            loss = criterion(preds_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"📊 Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "phase2_trained.pth")
    print("✅ Model saved to phase2_trained.pth")

if __name__ == "__main__":
    train_model(
        trace_path="data/traces/cnn_kernel_traces.json",
        model_path="generator/data/cnn_models.json"
    )

# This script trains a sequence-to-sequence model to predict CNN layer sequences
# from kernel traces, using the Phase2Model defined in phase2/model.py.
# It loads the OPi data and CNN model labels, trains the model, and saves the trained weights.
# It uses CrossEntropyLoss for training, ignoring padding tokens.
# The model is trained for a specified number of epochs with Adam optimizer.
# The training progress is printed to the console.
# The trained model is saved to "phase2_trained.pth".
# The script can be run directly to start the training process.
# It uses the `load_dataset` function from phase1.utils to load OPi data,
# and the `load_labels` function from phase2.utils to load CNN model labels.
# The model is trained on batches of data, and the average loss per epoch is reported.
# The model's state dictionary is saved at the end of training for later use.
# The training process can be customized with parameters like number of epochs,
# batch size, and learning rate.
# The device is set to "cuda" if available, otherwise "cpu".
# This script is part of the layer sequence extraction project, specifically for Phase 2.
# It assumes the necessary data files are present in the specified paths.
