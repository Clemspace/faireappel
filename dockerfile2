FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Download and cache models at build time
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('TheBloke/falcon-7b-instruct-GPTQ'); \
    AutoModelForCausalLM.from_pretrained('TheBloke/falcon-7b-instruct-GPTQ', device_map='auto', trust_remote_code=True)"

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "main2.py", "--server.port=8501", "--server.address=0.0.0.0"]