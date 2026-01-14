# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (if any specific C libraries are needed for numpy/pandas)
# netcat is often useful for health checks, but optional.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
# This copies infer.py, app.py, and the src/ directory
COPY . .

# 7. Set Environment Variables
# Ensures output is logged immediately (useful for debugging)
ENV PYTHONUNBUFFERED=1
# Port that FastAPI will listen on
ENV PORT=8000

# 8. Expose the port
EXPOSE 8000

# 9. Run the application
# We use shell form to allow variable expansion if needed, but array form is safer.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]