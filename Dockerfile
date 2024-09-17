# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files to the container
COPY . .

# Expose only the port for Streamlit (8501)
EXPOSE 8501

# Run both FastAPI server (from server.py) and Streamlit client (from client.py) concurrently
CMD ["sh", "-c", "uvicorn app.server:app --host 127.0.0.1 --port 8000 & streamlit run app/client.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false"]
