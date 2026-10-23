# sps-genai

FastAPI server with a bigram text generator, spaCy word embeddings, a CIFAR10 image classifier (CNN, Assignment 2), and an MNIST digit generator (GAN, Assignment 3).

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
| GET | `/generate_digits?num_samples=16` | Generate handwritten digits with the GAN; returns a PNG grid (1-64 samples) |

### Examples

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "bigram", "length": 5}'

curl http://localhost:8000/embedding/cat

curl -X POST -F "file=@some_image.jpg" http://localhost:8000/classify
# {"predicted_class": "cat", "confidence": 0.864}

curl "http://localhost:8000/generate_digits?num_samples=16" -o digits.png
```

## CIFAR10 classifier

- Model: `CNN` in `helper_lib/model.py` (64x64x3 input, two conv/ReLU/max-pool blocks, FC 100, FC 10).
- Training: `uv run python train_cifar10.py` downloads CIFAR10 to `data/`, trains for 10 epochs with Adam (lr=0.001), and saves weights to `models/cnn_cifar10.pth`. Test accuracy: 65.23%.
- Images are resized to 64x64 and normalized with the CIFAR10 mean/std, both in training and in the API (`helper_lib/data_loader.py::get_transform`).
- `uv run python test_classify.py` sends CIFAR10 test images to a running server's `/classify` endpoint.

## MNIST GAN

- Model: `GAN` in `helper_lib/model.py` (`Generator` + `Discriminator`).
  - Generator: noise (N, 100) -> FC to 7x7x128 -> ConvTranspose 128->64 (14x14) + BatchNorm + ReLU -> ConvTranspose 64->1 (28x28) + Tanh.
  - Discriminator: (1, 28, 28) -> Conv 1->64 (14x14) + LeakyReLU(0.2) -> Conv 64->128 (7x7) + BatchNorm + LeakyReLU(0.2) -> Flatten -> Linear -> Sigmoid.
- Training: `uv run python train_gan_mnist.py` downloads MNIST to `data/`, trains for 20 epochs with BCE loss and Adam (lr=2e-4, betas=(0.5, 0.999)) via `helper_lib/trainer.py::train_gan`, and saves weights to `models/gan_mnist.pth`.
- `helper_lib/generator.py::generate_samples` plots a grid of generated digits.
