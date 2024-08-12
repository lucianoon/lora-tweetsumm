# ── Build stage ────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only dependency files first (for better caching)
COPY pyproject.toml .
COPY src/__init__.py src/__init__.py

# Install dependencies (cached layer unless pyproject.toml changes)
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir "gradio>=4.0" "sacremoses>=0.1"

# ── Runtime stage ──────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Install the project in editable mode (lightweight, no deps needed)
RUN pip install --no-cache-dir --no-deps -e .

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# Pre-download T5 to cache the main demo model inside the container image.
# Optional PT↔EN translation models are downloaded only if translation is enabled.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
               AutoTokenizer.from_pretrained('google-t5/t5-small'); \
               AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-small')"

# Expose Gradio port
EXPOSE 7860

# Default: launch the demo
# Override with: docker run <image> python -m scripts.train
CMD ["python", "-m", "scripts.demo", "--port", "7860", "--allow-untrained"]
