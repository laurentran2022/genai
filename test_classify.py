"""Send a few CIFAR10 test images to the running API's /classify endpoint.

Usage: start the API (uv run fastapi dev app/main.py), then run
    uv run python test_classify.py [--url http://127.0.0.1:8000] [--count 10]
"""
import argparse
import io

import requests
from torchvision import datasets

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--count", type=int, default=10)
args = parser.parse_args()

# Raw 32x32 PIL images; the API applies the resize/normalize itself
test_set = datasets.CIFAR10(root="data", train=False, download=True)

correct = 0
for i in range(args.count):
    image, label = test_set[i]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = requests.post(
        f"{args.url}/classify",
        files={"file": (f"test_{i}.png", buffer.getvalue(), "image/png")},
    )
    response.raise_for_status()
    result = response.json()
    correct += result["predicted_class"] == CLASSES[label]
    print(f"true={CLASSES[label]:<10} predicted={result['predicted_class']:<10} confidence={result['confidence']:.3f}")

print(f"{correct}/{args.count} correct")
