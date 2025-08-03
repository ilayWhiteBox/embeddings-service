# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

# Copy shared and service-specific code
COPY main.py ./main.py

CMD ["gunicorn", "main:app", "--bind=0.0.0.0:8080", "--workers=1", "--threads=1", "--preload"]
