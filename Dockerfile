FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       git \
       libopenblas-dev \
       liblapack-dev \
       libjpeg-dev \
       libpng-dev \
       libx11-dev \
       libxrender-dev \
       libgtk-3-dev \
       libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . /app

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000

# Use the shell form so `$PORT` (set by Railway) is expanded at container start.
# Provide a default of 5000 when `PORT` is not set.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]
