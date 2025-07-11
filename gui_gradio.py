# gui_gradio.py

import gradio as gr
import torch
import matplotlib.pyplot as plt
from phase2.model import Phase2Model
from phase2.utils import decode_sequence, load_labels, VOCAB_SIZE, IDX_TO_TOKEN
from phase1.infer import infer_opi
from utils.evaluation import evaluate
from collections import Counter

def load_model(weights_path, input_dim=10, hidden_dim=128):
    model = Phase2Model(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict_and_evaluate(trace_path, model_path, weight_path, n_samples):
    try:
        model = load_model(weight_path)
        X = infer_opi(trace_path, model_path, weights="phase1_trained.pth", max_len=64)
        labels = load_labels(model_path, max_len=50)

        preds = model(X).argmax(dim=-1).tolist()
        labels = labels.tolist()

        preds_trim = [[x for x in seq if x != 10 and x != 11] for seq in preds]
        labels_trim = [[x for x in seq if x != 10 and x != 11] for seq in labels]

        decoded_preds = [decode_sequence(seq) for seq in preds_trim[:n_samples]]
        decoded_trues = [decode_sequence(seq) for seq in labels_trim[:n_samples]]

        sample_display = ""
        for i in range(min(n_samples, len(decoded_preds))):
            sample_display += f"🔹 Sample {i+1}\n"
            sample_display += f"Predicted    : {decoded_preds[i]}\n"
            sample_display += f"Ground Truth : {decoded_trues[i]}\n"
            sample_display += "-"*50 + "\n"

        metrics = evaluate(preds, labels)
        metric_str = f"📊 Evaluation:\nLER: {metrics['LER']:.4f} | F1: {metrics['F1']:.4f} | MAE: {metrics['MAE']:.4f}"

        with open("predictions.txt", "w") as f:
            f.write(sample_display)

        fig = plot_distribution_chart(preds_trim[:n_samples], labels_trim[:n_samples])
        return sample_display, metric_str, "✅ predictions.txt saved", fig

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", None

def plot_distribution_chart(preds, labels):
    pred_tokens = [IDX_TO_TOKEN[x] for seq in preds for x in seq if x in IDX_TO_TOKEN]
    true_tokens = [IDX_TO_TOKEN[x] for seq in labels for x in seq if x in IDX_TO_TOKEN]

    pred_counts = Counter(pred_tokens)
    true_counts = Counter(true_tokens)

    all_keys = sorted(set(pred_counts.keys()).union(true_counts.keys()))
    pred_vals = [pred_counts.get(k, 0) for k in all_keys]
    true_vals = [true_counts.get(k, 0) for k in all_keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(all_keys, pred_vals, alpha=0.6, label="Predicted")
    ax.bar(all_keys, true_vals, alpha=0.6, label="Ground Truth")
    ax.set_ylabel("Count")
    ax.set_title("Layer Token Distribution")
    ax.legend()
    return fig

iface = gr.Interface(
    fn=predict_and_evaluate,
    inputs=[
        gr.Textbox(label="Trace Path", value="data/traces/cnn_kernel_traces.json"),
        gr.Textbox(label="Model JSON Path", value="generator/data/cnn_models_optimized.json"),
        gr.Textbox(label="Phase2 Weights Path", value="phase2_trained.pth"),
        gr.Slider(1, 20, value=5, label="Number of Samples to View")
    ],
    outputs=[
        gr.Textbox(label="🧾 Sample Predictions"),
        gr.Textbox(label="📊 Evaluation Metrics"),
        gr.Textbox(label="📂 Status"),
        gr.Plot(label="📈 Token Distribution")
    ],
    title="🔍 Layer Sequence Extractor",
    description="Compare predicted CNN layer sequences using traces and Phase 2 model. Outputs prediction, metrics & visualizations.",
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()
# gui_gradio.py
    print("Gradio interface launched. Open the provided URL to interact with the model.")