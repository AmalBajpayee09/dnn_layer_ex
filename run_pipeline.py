from phase1.train import train_model as train_phase1
from phase2.train import train_model as train_phase2
from phase2.predict import predict
from utils.evaluation import evaluate
from optimization.optimize_with_tvm import optimize_model
from phase2.utils import decode_sequence  # ⬅️ ADD THIS

import os

TRACE_PATH = "data/traces/cnn_kernel_traces.json"
MODEL_PATH = "generator/data/cnn_models.json"
OPTIMIZED_MODEL_PATH = "generator/data/cnn_models_optimized.json"

print("🚀 Step 1: Training Phase 1 model...")
train_phase1(TRACE_PATH, MODEL_PATH)

print("🛠️ Step 2: Optimizing CNN models using TVM...")
optimize_model(input_path=MODEL_PATH, output_path=OPTIMIZED_MODEL_PATH)

print("🚀 Step 3: Training Phase 2 model using optimized model traces...")
train_phase2(TRACE_PATH, OPTIMIZED_MODEL_PATH)

print("🔎 Step 4: Running prediction on optimized models...")
preds, labels = predict(TRACE_PATH, OPTIMIZED_MODEL_PATH, weights="phase2_trained.pth")

# 🔍 Print Ground Truth vs Predicted for 5 samples
print("\n🧾 Sample Predictions (first 5):")
for i in range(min(5, len(preds))):
    print(f"\n🔹 Sample {i+1}")
    print("Predicted    :", decode_sequence(preds[i]))
    print("Ground Truth :", decode_sequence(labels[i]))
    print("-" * 60)

print("\n📊 Step 5: Evaluation Results:")
results = evaluate(preds, labels)
for k, v in results.items():
    print(f"{k}: {v:.4f}")
print("✅ Pipeline completed successfully!")
# Save predictions to file
with open("predictions.txt", "w") as f:
    for pred, label in zip(preds, labels):
        f.write(f"Predicted: {decode_sequence(pred)} | True: {decode_sequence(label)}\n")
print("✅ Predictions saved to predictions.txt")