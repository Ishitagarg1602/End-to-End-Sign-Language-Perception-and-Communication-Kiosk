# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by MediaPipe and OpenCV
# libgles2-mesa is specifically required to fix the libGLESv2.so.2 error
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgles2-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r mvp/backend/requirements.txt

# Make sure the container listens on the port Render provides
EXPOSE 10000

# Command to run the application
CMD ["sh", "-c", "uvicorn mvp.backend.main:fastapi_app --host 0.0.0.0 --port ${PORT:-10000}"]
