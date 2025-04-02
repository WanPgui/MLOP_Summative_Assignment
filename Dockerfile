# Use a lightweight Python image
FROM python:3.11-slim

# Install build dependencies and SQLite
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libxft-dev \
    libpng-dev \
    libharfbuzz-dev \
    libpango1.0-dev \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy current directory contents into the container
COPY . .

# Convert CSV file to a database (if applicable)
COPY create_db.py /app/create_db.py  
RUN python /app/create_db.py || echo "Skipping DB creation"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Set environment variables to prevent Flask from exiting
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development 

# Start Flask when the container runs
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
