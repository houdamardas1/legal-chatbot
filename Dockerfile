FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image several GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

COPY app.py ./
COPY data ./data

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
CMD ["sh", "-c", "legal-chatbot build-index && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
