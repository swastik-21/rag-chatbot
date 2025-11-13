FROM python:3.11-slim

WORKDIR /app

# Upgrade pip and install build tools first (cached layer)
RUN pip install --upgrade pip setuptools wheel

# Install Poetry (cached layer)
RUN pip install poetry && \
    poetry config virtualenvs.create false

# Copy dependency files first (cached layer if dependencies don't change)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code last (changes most frequently)
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

