# run_pipeline.py

from phase1.train import train_model as train_phase1
from phase2.train import train_model as train_phase2
from phase2.predict import predict
from utils.evaluation import compute_metrics, decode_sequence
from optimization.optimize_with_tvm import optimize_model

import os

# 📁 Paths
TRACE_PATH = "data/traces/cnn_kernel_traces.json"
MODEL_PATH = "generator/data/cnn_models.json"
OPTIMIZED_MODEL_PATH = "generator/data/cnn_models_optimized.json"

# 🚀 Step 1: Train Phase 1 (OPi Prediction)
print("🚀 Step 1: Training Phase 1 model...")
train_phase1(TRACE_PATH, MODEL_PATH)

# ⚙️ Step 2: Optimize models using TVM
print("🛠️ Step 2: Optimizing CNN models using TVM...")
optimize_model(input_path=MODEL_PATH, output_path=OPTIMIZED_MODEL_PATH)

# 🚀 Step 3: Train Phase 2 (Layer Sequence Prediction)
print("🚀 Step 3: Training Phase 2 model using optimized model traces...")
train_phase2(TRACE_PATH, OPTIMIZED_MODEL_PATH)

# 🔍 Step 4: Predict Layer Sequences
print("🔎 Step 4: Running prediction on optimized models...")
preds, labels = predict(TRACE_PATH, OPTIMIZED_MODEL_PATH, weights="phase2_trained.pth")

# 📤 Save predictions
with open("predictions.txt", "w") as f:
    for pred, label in zip(preds, labels):
        f.write(f"Predicted: {decode_sequence(pred)} | True: {decode_sequence(label)}\n")
print("✅ Predictions saved to predictions.txt")

# 🧪 Step 5: Evaluation
print("\n📊 Step 5: Evaluation Results:")
ler, f1 = compute_metrics(preds, labels)
print(f"Layer Error Rate (LER): {ler:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# 💾 Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Layer Error Rate (LER): {ler:.2f}%\nF1 Score: {f1:.2f}%\n")
print("✅ Metrics saved to metrics.txt")

# 👀 Display sample predictions
print("\n🧾 Sample Predictions (first 5):")
for i in range(min(5, len(preds))):
    print(f"\n🔹 Sample {i+1}")
    print("Predicted    :", decode_sequence(preds[i]))
    print("Ground Truth :", decode_sequence(labels[i]))
    print("-" * 60)

print("\n🎉 All steps completed successfully!")
