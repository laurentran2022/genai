# Use an official Python runtime
FROM python:3.12-slim-bookworm

# Install curl and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /code

# Copy dependency files
COPY pyproject.toml uv.lock /code/

# Install project dependencies
RUN uv sync --frozen

# Install the spaCy model used by the assignment
RUN uv run python -m spacy download en_core_web_lg

# Copy application code
COPY ./app /code/app

# Run FastAPI inside the container on port 80
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
