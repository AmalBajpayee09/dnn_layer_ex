# optimize_with_tvm.py

import json
import os
import torch
from torchvision.models import resnet18
from tvm import relay
import tvm

def optimize_model(input_path=None, output_path=None):
    """
    Optimizes CNN models using TVM.
    - If input_path is None: optimize a demo ResNet18 model.
    - Else: load CNN JSON model list and (mock) optimize them.
    """

    if input_path is None:
        # 🔧 TVM Optimization for ResNet18 (demo only)
        model = resnet18(pretrained=False)
        model.eval()

        input_data = torch.randn(1, 3, 224, 224)
        scripted_model = torch.jit.trace(model, input_data)
        shape_list = [("input", input_data.shape)]

        # 📦 Convert PyTorch to TVM Relay
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # 🛠️ Compile with TVM
        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        print("✅ Optimized ResNet18 using TVM (demo)")
        return

    # 📂 Optimize (mock) JSON CNN models
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input model file not found: {input_path}")

    with open(input_path, "r") as f:
        models = json.load(f)

    # 🔄 Placeholder: TVM compile per model if needed
    # This is a mock optimization loop

    if output_path is None:
        output_path = input_path.replace(".json", "_optimized.json")

    with open(output_path, "w") as f:
        json.dump(models, f, indent=2)

    print(f"✅ Mock optimized models saved → {output_path}")


if __name__ == "__main__":
    optimize_model(
        input_path="generator/data/cnn_models.json",
        output_path="generator/data/cnn_models_optimized.json"
    )
