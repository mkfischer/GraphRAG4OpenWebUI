# Use official Python base image with explicit architecture specification
FROM --platform=linux/amd64 python:3.10-slim AS build-stage

# Install system dependencies needed for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    cython3 \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Update PATH to include Cargo binaries
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin via cargo
RUN cargo install maturin

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install build-time dependencies (numpy and Cython)
RUN pip install --no-cache-dir numpy Cython

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt --no-build-isolation

# Stage 2: Final stage with only necessary artifacts
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV API_BASE=http://host.containers.internal:11434/v1
ENV GRAPHRAG_API_KEY=your_actual_api_key_here
ENV GRAPHRAG_API_KEY_EMBEDDING=your_embedding_api_key_here
ENV GRAPHRAG_LLM_MODEL=gemma2
ENV API_BASE_EMBEDDING=https://api.openai.com/v1
ENV GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
ENV INPUT_DIR=/app/inputs/artifacts

# Set the working directory in the container
WORKDIR /app

# Copy the installed Python packages from the build stage
COPY --from=build-stage /install /usr/local

# Copy the rest of the application code
COPY . .

# Expose port 8012
EXPOSE 8012

# Run main.py when the container launches
CMD ["python", "main.py"]
