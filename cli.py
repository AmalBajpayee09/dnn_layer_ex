import argparse
from phase1.train import train_model as train_phase1
from phase2.train import train_model as train_phase2
from phase2.predict import predict
from utils.evaluation import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2"], required=True)
    parser.add_argument("--task", choices=["train", "predict", "evaluate"], required=True)
    parser.add_argument("--trace", default="data/traces/cnn_kernel_traces.json")
    parser.add_argument("--model", default="generator/data/cnn_models.json")
    parser.add_argument("--weights", default="utils/phase2_trained.pth")
    args = parser.parse_args()

    if args.phase == "1" and args.task == "train":
        train_phase1(args.trace, args.model)

    elif args.phase == "2":
        if args.task == "train":
            train_phase2(args.trace, args.model)
        elif args.task == "predict":
            preds, labels = predict(args.trace, args.model, weights=args.weights)
            for i, (p, t) in enumerate(zip(preds, labels)):
                print(f"🔢 Sample {i+1}")
                print("🔮 Pred:", p)
                print("🎯 True:", t)
                if i == 4:
                    break
        elif args.task == "evaluate":
            preds, labels = predict(args.trace, args.model, weights=args.weights)
            result = evaluate(preds, labels)
            print("📊 Evaluation:")
            for k, v in result.items():
                print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
