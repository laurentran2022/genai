import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from bigram_model import BigramModel
import spacy
import torch
from torchvision.utils import save_image

# Make the project root importable so the API can use helper_lib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from helper_lib.data_loader import get_transform
from helper_lib.model import get_model
from helper_lib.utils import load_model

app = FastAPI()
nlp = spacy.load("en_core_web_lg")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Load the CIFAR10 classifier once at startup
cnn_model = load_model(get_model("CNN"), PROJECT_ROOT / "models" / "cnn_cifar10.pth")
cnn_transform = get_transform()

# Load the MNIST GAN once at startup
gan_model = load_model(get_model("GAN"), PROJECT_ROOT / "models" / "gan_mnist.pth")
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}

@app.get("/embedding/{word}")
def get_embedding(word: str):
    embedding = nlp(word).vector
    return {
        "word": word,
        "embedding": embedding.tolist()
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    batch = cnn_transform(image).unsqueeze(0)  # Shape: [1, 3, 64, 64]
    with torch.no_grad():
        probabilities = torch.softmax(cnn_model(batch), dim=1)[0]
    confidence, index = probabilities.max(dim=0)

    return {
        "predicted_class": CIFAR10_CLASSES[index.item()],
        "confidence": round(confidence.item(), 4),
    }


@app.get("/generate_digits", response_class=Response)
def generate_digits(num_samples: int = Query(16, ge=1, le=64)):
    """Generate handwritten digits with the MNIST GAN and return them as a PNG grid."""
    with torch.no_grad():
        noise = torch.randn(num_samples, gan_model.z_dim)
        images = gan_model.generator(noise) * 0.5 + 0.5  # Undo the [-1, 1] normalization

    buffer = io.BytesIO()
    save_image(images, buffer, format="png", nrow=min(num_samples, 8), padding=2)
    return Response(content=buffer.getvalue(), media_type="image/png")
