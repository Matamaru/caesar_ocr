FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY caesar_ocr /app/caesar_ocr
# Models are downloaded from S3 at runtime; keep the image small.

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[api]

EXPOSE 8000

CMD ["uvicorn", "caesar_ocr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
