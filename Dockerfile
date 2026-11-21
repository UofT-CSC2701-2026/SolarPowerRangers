FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY hourly_forecast.py solar_xgboost_model.json ./
COPY web/ ./web/

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
