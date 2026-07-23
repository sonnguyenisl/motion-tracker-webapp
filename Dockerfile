FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Shell form so ${PORT} (provided by Railway) is expanded. Bind to 0.0.0.0 so
# the container is reachable. Single worker (-w 1) is required: SocketIO rooms
# and the background scoring task live in one process.
# --max-requests / --max-requests-jitter: Kill and auto-restart the worker
# every ~6-8 requests so leaked memory (numpy, OpenCV, vector cache) is
# reclaimed by the OS instead of accumulating until OOM.
CMD gunicorn -w 1 --threads 4 \
  --bind 0.0.0.0:${PORT:-8080} run:app
