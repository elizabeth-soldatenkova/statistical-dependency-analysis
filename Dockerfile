FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

FROM base AS production

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test

COPY requirements-test.txt .

RUN pip install --no-cache-dir -r requirements-test.txt

COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "tests/test_api.py", "-v"]
