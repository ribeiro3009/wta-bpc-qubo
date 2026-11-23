FROM python:3.13-slim

WORKDIR /app

# numpy (OpenBLAS) would start one thread per CPU even for tiny matrices: ~6x slower here
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# Install the project and its dependencies (numpy, scipy, highspy, pytest, qiskit)
COPY . .
RUN pip install --no-cache-dir -e ".[dev,quantum]"

# Default command: run the tests
CMD ["python", "-m", "pytest", "-q"]
