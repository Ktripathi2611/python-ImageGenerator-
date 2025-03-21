FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_WATCH_DIRS=false

# Expose port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType=none", "--server.port=8501", "--server.address=0.0.0.0"] 