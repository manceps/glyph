FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcairo2-dev \
        pkg-config \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY tests/ tests/

EXPOSE 8090

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090"]
