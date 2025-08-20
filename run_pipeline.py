# run_pipeline.py

from phase1.train import train_model as train_phase1
from phase2.train import train_model as train_phase2
from phase2.predict import predict
from utils.evaluation import compute_metrics, decode_sequence
from optimization.optimize_with_tvm import optimize_model


import os

# 📁 File Paths
TRACE_PATH = "data/traces/cnn_kernel_traces.json"
MODEL_PATH = "generator/data/cnn_models.json"
OPTIMIZED_MODEL_PATH = "generator/data/cnn_models_optimized.json"
PHASE1_WEIGHTS = "phase1_trained.pth"
PHASE2_WEIGHTS = "phase2_trained.pth"

# 🚀 Step 1: Train Phase 1 - OPi Prediction
print("\n🚀 Step 1: Training Phase 1 model...")
train_phase1(TRACE_PATH, MODEL_PATH, save_path=PHASE1_WEIGHTS)

# ⚙️ Step 2: Optimize Models with TVM
print("\n🛠️ Step 2: Optimizing CNN models using TVM...")
optimize_model(input_path=MODEL_PATH, output_path=OPTIMIZED_MODEL_PATH)

# 🚀 Step 3: Train Phase 2 - Layer Sequence Prediction
print("\n🚀 Step 3: Training Phase 2 model using optimized traces...")
train_phase2(trace_path=TRACE_PATH, model_path=OPTIMIZED_MODEL_PATH)


# 🔍 Step 4: Inference - Predict Layer Sequences
print("\n🔎 Step 4: Running predictions on optimized models...")
preds, labels = predict(TRACE_PATH, OPTIMIZED_MODEL_PATH, weights=PHASE2_WEIGHTS)

# 📤 Step 5: Save Predictions to File
with open("predictions.txt", "w") as f:
    for pred, label in zip(preds, labels):
        f.write(f"Predicted: {decode_sequence(pred)} | True: {decode_sequence(label)}\n")
print("✅ Predictions saved to predictions.txt")

# 📊 Step 6: Evaluation Metrics
print("\n📊 Step 5: Evaluation Results:")
ler, f1 = compute_metrics(preds, labels)
print(f"Layer Error Rate (LER): {ler:.2f}%")
print(f"F1 Score: {f1:.2f}%")

with open("metrics.txt", "w") as f:
    f.write(f"Layer Error Rate (LER): {ler:.2f}%\nF1 Score: {f1:.2f}%\n")
print("✅ Metrics saved to metrics.txt")

# 👀 Step 7: Sample Predictions Preview
print("\n🧾 Sample Predictions (first 5):")
for i in range(min(5, len(preds))):
    print(f"\n🔹 Sample {i+1}")
    print("Predicted    :", decode_sequence(preds[i]))
    print("Ground Truth :", decode_sequence(labels[i]))
    print("-" * 60)

print("\n🎉 All steps completed successfully!")
