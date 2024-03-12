# FROM python:3.10.13

# WORKDIR /app

# COPY . /app

# # Install specific versions of dependencies
# RUN pip install tensorflow-cpu flask flask_cors pillow matplotlib opencv-python-headless

# # Expose port 5000 (adjust the port number as needed)
# EXPOSE 5000

# # Run the Flask application
# CMD ["python3", "app.py"]
# Stage 1: Build the application
FROM python:3.10.13-slim as builder

# Set working directory
WORKDIR /app

# Copy the entire application code
COPY . .

# Stage 2: Create the final runtime image
FROM python:3.10.13-slim

# Set working directory
WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /app .

# Install only necessary dependencies
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir tensorflow-cpu==2.15.0 flask flask_cors pillow matplotlib opencv-python-headless


# Expose port 5000
EXPOSE 5000

# Cleanup unnecessary files
RUN apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache \
    && rm -rf /tmp/*

# Command to run the application
CMD ["python3", "app.py"]