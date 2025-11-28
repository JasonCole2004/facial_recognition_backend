FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies for DeepFace + OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (add more if needed)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port FastAPI will run on
EXPOSE 8000

# Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]