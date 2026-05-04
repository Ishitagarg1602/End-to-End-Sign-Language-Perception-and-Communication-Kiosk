# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

<<<<<<< HEAD
# Install system dependencies required by MediaPipe and OpenCV
=======
# Install system dependencies required by MediaPipe, OpenCV, and Whisper (ffmpeg)
>>>>>>> 6d29a844d173a8e5dbdcaef04d30b440e24fdd5a
# libgles2-mesa is specifically required to fix the libGLESv2.so.2 error
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
<<<<<<< HEAD
=======
    ffmpeg \
>>>>>>> 6d29a844d173a8e5dbdcaef04d30b440e24fdd5a
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r mvp/backend/requirements.txt

<<<<<<< HEAD
# Make sure the container listens on the port Render provides
EXPOSE 10000

# Command to run the application
CMD ["sh", "-c", "uvicorn mvp.backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
=======
# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Command to run the application on port 7860 for Hugging Face Spaces
CMD ["sh", "-c", "uvicorn mvp.backend.main:app --host 0.0.0.0 --port 7860"]
>>>>>>> 6d29a844d173a8e5dbdcaef04d30b440e24fdd5a
