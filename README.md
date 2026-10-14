# sps-genai

FastAPI server with a bigram text generator, spaCy word embeddings, and a CIFAR10 image classifier (CNN, Assignment 2).

## Run with Docker

```bash
git clone https://github.com/laurentran2022/genai.git
cd genai
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

The API is then available at http://localhost:8000 (interactive docs at http://localhost:8000/docs).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/generate` | Bigram text generation. Body: `{"start_word": "bigram", "length": 5}` |
| GET | `/embedding/{word}` | spaCy word embedding |
| POST | `/classify` | Upload an image; returns the predicted CIFAR10 class and confidence |

### Examples

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "bigram", "length": 5}'

curl http://localhost:8000/embedding/cat

curl -X POST -F "file=@some_image.jpg" http://localhost:8000/classify
# {"predicted_class": "cat", "confidence": 0.864}
```

## CIFAR10 classifier

- Model: `CNN` in `helper_lib/model.py` (64x64x3 input, two conv/ReLU/max-pool blocks, FC 100, FC 10).
- Training: `uv run python train_cifar10.py` downloads CIFAR10 to `data/`, trains for 10 epochs with Adam (lr=0.001), and saves weights to `models/cnn_cifar10.pth`. Test accuracy: 65.23%.
- Images are resized to 64x64 and normalized with the CIFAR10 mean/std, both in training and in the API (`helper_lib/data_loader.py::get_transform`).
- `uv run python test_classify.py` sends CIFAR10 test images to a running server's `/classify` endpoint.
