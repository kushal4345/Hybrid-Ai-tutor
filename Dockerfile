# Final Dockerfile with Debugging

# STEP 1: Show us exactly what files are in the build directory.
# This will solve the mystery.
RUN ls -R

# STEP 2: Start from a standard Python 3.11 image.
FROM python:3.11-slim

# STEP 3: Set the working directory inside the container.
WORKDIR /app

# STEP 4: Install system libraries needed for PyMuPDF (fitz).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    mupdf \
    && rm -rf /var/lib/apt/lists/*

# STEP 5: Copy the requirements file into the container.
COPY requirements.txt .

# STEP 6: Install all the Python libraries.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# STEP 7: Copy your main application file into the container.
COPY main.py .

# STEP 8: Expose the port your app runs on.
EXPOSE 8000

# STEP 9: The command to start your server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]```

#### **Step 3: Check Your Railway Settings**

1.  Go to your `Hybrid-Ai-tutor` service in Railway.
2.  Go to the **"Settings"** tab.
3.  Under the **"Build"** section:
    *   Make sure **Builder** is set to **`Dockerfile`**.
    *   Make sure **Root Directory** is **empty** or a single slash **`/`**.

#### **Step 4: Commit and Push**

1.  Save all your changes.
2.  Commit and push to GitHub.
    ```bash
    git add .
    git commit -m "fix: final attempt with Dockerfile and debug step"
    git push origin main
    ```

### **How to Read the New Error Log**

This will cause a new build on Railway. It will likely fail again, but this time, it will give us the answer.

1.  Go to the new, failed deployment.
2.  Click **"View Logs"**.
3.  Scroll up to the very top of the build log.

You will see the output of the `RUN ls -R` command from Step 1 of the Dockerfile. It will look something like this: