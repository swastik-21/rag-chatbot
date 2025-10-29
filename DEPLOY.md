# Deployment Guide

## Local Development

1. Install dependencies:
```bash
poetry install
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Start the application:
```bash
./start_chatbot.sh
```

## Docker Deployment

Build and run with Docker:
```bash
docker build -t shopilots-chatbot .
docker run -p 8080:8080 -e OPENAI_API_KEY="your-api-key" shopilots-chatbot
```

## Heroku Deployment

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Set environment variable: `heroku config:set OPENAI_API_KEY="your-api-key"`
5. Push: `git push heroku main`

## Railway Deployment

1. Connect your GitHub repository to Railway
2. Set environment variable `OPENAI_API_KEY` in Railway dashboard
3. Deploy automatically on push

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI API access

## Notes

- The vector database (`vector_store/`) is persisted locally
- For production, you may want to use a managed vector database service
- Ensure vector database is built before first deployment: `python chatbot/memory_builder.py`
