# Dockerfile

# Start from a standard, lightweight Python 3.11 image.
FROM python:3.11-slim

# Set the working directory inside the container to /app.
WORKDIR /app

# Install the system-level libraries that PyMuPDF (fitz) needs BEFORE installing Python packages.
# This is the most critical step.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the requirements file into the container first.
# This is a Docker optimization technique.
COPY app/requirements.txt .

# Now, run pip to install all the Python libraries from the requirements file.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (everything in the 'app' folder) into the container.
COPY ./app .

# Tell Railway that your application will listen on port 8000.
EXPOSE 8000

# The final command to start your Uvicorn server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]