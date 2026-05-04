# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by MediaPipe, OpenCV, and Whisper (ffmpeg)
# libgles2-mesa is specifically required to fix the libGLESv2.so.2 error
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    ffmpeg \
    portaudio19-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Command to run the application on port 7860 for Hugging Face Spaces
CMD ["sh", "-c", "PYTHONPATH=/app/backend uvicorn backend.main:app --host 0.0.0.0 --port 7860"]
