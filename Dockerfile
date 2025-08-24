# Final Dockerfile

# Start from a standard Python 3.11 image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Install system libraries needed for PyMuPDF (fitz).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container.
COPY requirements.txt .

# Install all the Python libraries.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your main application file into the container.
COPY main.py .

# Expose the port your app runs on.
EXPOSE 8000

# The command to start your server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]