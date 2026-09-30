FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck: agent process responds to --help-style import check.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import src; print('ok')" || exit 1

CMD ["python", "main.py"]
