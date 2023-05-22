FROM python:latest

WORKDIR /app

ENV PORT 8000
ENV HOST 0.0.0.0

# Copy only the requirements file first
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
