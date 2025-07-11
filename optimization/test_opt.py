# optimization/test_opt.py

from optimize_with_tvm import optimize_model
import torchvision.models as models

if __name__ == "__main__":
    optimize_model(models.resnet18)
