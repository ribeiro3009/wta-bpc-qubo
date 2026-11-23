FROM python:3.13-slim

WORKDIR /app

# Install the project and its dependencies (numpy, scipy, highspy, pytest)
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default command: run the tests
CMD ["python", "-m", "pytest", "-q"]
