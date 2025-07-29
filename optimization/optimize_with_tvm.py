# optimize_with_tvm.py

import json
import os
import torch
from torchvision.models import resnet18
from tvm import relay, runtime
import tvm

def optimize_model(input_path=None, output_path=None):
    """
    Optimizes CNN models using TVM. If no input_path is given, defaults to ResNet18.
    Otherwise, loads CNN models from a JSON and re-saves (mock optimized).
    """
    
    if input_path is None:
        # 🔧 Optimize ResNet18 using TVM for testing
        model = resnet18(pretrained=False)
        model.eval()

        input_data = torch.randn(1, 3, 224, 224)
        scripted_model = torch.jit.trace(model, input_data)
        shape_list = [("input", input_data.shape)]

        # 📦 Convert to Relay
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # 🛠️ Compile with TVM
        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        print("✅ Optimized resnet18 using TVM (demo mode)")
        return

    # 📂 Optimize (mock) CNN JSON model structure
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model file not found: {input_path}")

    with open(input_path, "r") as f:
        models = json.load(f)

    # 🔄 Place for TVM per-model optimization (future scope)

    # 💾 Save as optimized
    if output_path is None:
        output_path = input_path.replace(".json", "_optimized.json")

    with open(output_path, "w") as f:
        json.dump(models, f, indent=2)

    print(f"✅ Models re-saved as optimized → {output_path}")


if __name__ == "__main__":
    optimize_model(
        input_path="generator/data/cnn_models.json",
        output_path="generator/data/cnn_models_optimized.json"
    )
