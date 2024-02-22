# Stage 1: Build Python environment with dependencies
FROM python:3.9 AS builder

WORKDIR /app

COPY requirements.txt .

RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Create the final Docker image
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /app/venv /app/venv
COPY . .

# Specify the Python interpreter to use
ENV PATH="/app/venv/bin:$PATH"

# Your additional Docker configuration commands go here

# Command to run Flask application
CMD ["python", "app.py"]
