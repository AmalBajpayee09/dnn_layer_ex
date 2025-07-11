# generator/generate_dnn_dataset.py
import json
import random
from pathlib import Path

CNN_OUTPUT = Path("data/cnn_models.json")
RNN_OUTPUT = Path("data/rnn_models.json")

def generate_cnn_models(num_models=4000):
    cnn_models = []
    for _ in range(num_models):
        model = {"layers": []}
        num_cnn_layers = random.randint(5, 20)
        num_fcn_layers = random.randint(1, 5)

        for _ in range(num_cnn_layers):
            layer_type = random.choices(["conv2d", "resnet", "mobilenet"], weights=[0.6, 0.2, 0.2])[0]
            in_ch = random.choice([3, 16, 32, 64])
            out_ch = random.choice([16, 32, 64, 128])
            kernel = random.choice([1, 3, 5])
            model["layers"].append({
                "type": layer_type,
                "in_channels": in_ch,
                "out_channels": out_ch,
                "kernel_size": kernel
            })
            # Optional layers
            if random.random() < 0.5:
                model["layers"].append({"type": random.choice(["batchnorm", "relu", "tanh", "sigmoid"])})
            if random.random() < 0.3:
                model["layers"].append({"type": "pool", "kernel_size": 2})

        # Add FC layers at the end
        for _ in range(num_fcn_layers):
            model["layers"].append({"type": "fcn", "neurons": random.choice([64, 128, 256, 512])})
        model["layers"].append({"type": "softmax"})

        cnn_models.append(model)
    return cnn_models

def generate_rnn_models(num_models=4000):
    rnn_models = []
    for _ in range(num_models):
        model = {"layers": []}
        depth = random.randint(5, 50)
        remaining = depth

        while remaining > 0:
            layer_type = random.choice(["rnn", "gru", "lstm", "fcn"])
            out_dim = random.choice([4, 8, 16, 32, 64, 128])
            model["layers"].append({"type": layer_type, "out_dim": out_dim})

            if random.random() < 0.5:
                model["layers"][-1]["activation"] = random.choice(["tanh", "sigmoid"])
            remaining -= 1

        model["layers"].append({"type": "fcn", "neurons": 32, "activation": "sigmoid"})
        model["layers"].append({"type": "fcn", "neurons": 16, "activation": "sigmoid"})

        rnn_models.append(model)
    return rnn_models

def main():
    Path("data").mkdir(exist_ok=True)
    cnn_data = generate_cnn_models()
    rnn_data = generate_rnn_models()

    with open(CNN_OUTPUT, "w") as f:
        json.dump(cnn_data, f, indent=2)
    with open(RNN_OUTPUT, "w") as f:
        json.dump(rnn_data, f, indent=2)

    print(f"✅ Generated {len(cnn_data)} CNN models → {CNN_OUTPUT}")
    print(f"✅ Generated {len(rnn_data)} RNN models → {RNN_OUTPUT}")

if __name__ == "__main__":
    main()
