# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set environment variable to load model from MLflow
ENV LOAD_FROM_MLFLOW=true
ENV MLFLOW_RUN_ID=3c4f3ec2d48c449687f4b500a57e125a

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app (adjust if needed)
CMD ["uvicorn", "application.testing:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# After the creation of the Dockerfile, you can build the Docker image using the following command:
# docker build -t my-fastapi-app .
# Then, run the Docker container using:
# docker run -p 8000:8000 my-fastapi-app