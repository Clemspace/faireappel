FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make sure the entrypoint script is executable
RUN chmod +x Procfile

# Default port (will be overridden by Koyeb)
ENV PORT=8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health

# Let Koyeb handle the command through Procfile
CMD streamlit run main_test.py --server.port=$PORT --server.address=0.0.0.0