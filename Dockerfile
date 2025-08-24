# Final and Corrected Dockerfile

# STEP 1: Start from a standard Python 3.11 image. This is the build stage.
FROM python:3.11-slim

# STEP 2: Set the working directory inside the container.
WORKDIR /app

# STEP 3: Install system libraries needed for PyMuPDF (fitz).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf \
    && rm -rf /var/lib/apt/lists/*

# STEP 4: Copy the requirements file into the container.
COPY requirements.txt .

# STEP 5: Install all the Python libraries.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# STEP 6: Copy your main application file into the container.
COPY main.py .

# STEP 7: Expose the port your app runs on.
EXPOSE 8000

# STEP 8: The command to start your server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]