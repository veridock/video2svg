FROM python:3.9-slim

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Kopiuj pliki
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port dla web UI
EXPOSE 8000

# Uruchom aplikację
CMD ["python", "-m", "http.server", "8000", "--directory", "web"]