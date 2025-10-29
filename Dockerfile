FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
