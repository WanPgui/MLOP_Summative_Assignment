# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install build dependencies for compiling native modules
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libxft-dev \
    libpng-dev \
    libharfbuzz-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .
COPY diabetic_data.csv /app/diabetic_data.csv


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run the app
CMD ["python", "app.py"]
