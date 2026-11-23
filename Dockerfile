FROM python:3.13-slim

WORKDIR /app

# Install the project and its dependencies (numpy, scipy, highspy, pytest, qiskit)
COPY . .
RUN pip install --no-cache-dir -e ".[dev,quantum]"

# Default command: run the tests
CMD ["python", "-m", "pytest", "-q"]
