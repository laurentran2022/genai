from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import BigramModel
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_lg")
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
