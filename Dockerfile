# Use Python 3.11.8 base image
FROM python:3.11.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app from fastapi_ml_pipeline.py
CMD ["uvicorn", "fastapi_ml_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
