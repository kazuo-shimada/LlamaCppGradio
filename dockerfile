# 1) Base image
FROM python:3.10-slim

# 2) System deps (build tools + basic libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3) Create a non-root user (safer)
ARG APP_USER=appuser
RUN useradd -ms /bin/bash ${APP_USER}

# 4) Set working directory
WORKDIR /app

# 5) Copy project files first (only what’s needed)
# If you use a "src/" layout, copy it and the runner, plus metadata files
COPY requirements.txt ./
COPY main.py ./
COPY src ./src
# If you have pyproject.toml, uncomment next line:
# COPY pyproject.toml ./

# 6) Install Python deps
# Prefer no-cache-dir to keep image small
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# If you use a src/ package and want import by name "p3g", install it (editable optional):
# RUN pip install --no-cache-dir -e .

# 7) Expose the Gradio port
EXPOSE 7860

# 8) Environment defaults (can be overridden at runtime)
ENV MODEL_PATH=""
ENV MMPROJ_PATH=""

# 9) Use non-root user
USER ${APP_USER}

# 10) Entrypoint (start the app)
CMD ["python", "main.py"]
