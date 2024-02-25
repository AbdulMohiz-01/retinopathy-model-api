FROM python:3.10.13

WORKDIR /app

COPY . /app

# Install specific versions of dependencies
RUN pip install tensorflow-cpu flask flask_cors pillow matplotlib opencv-python-headless

# Expose port 5000 (adjust the port number as needed)
EXPOSE 5000

# Run the Flask application
CMD ["python3", "app.py"]