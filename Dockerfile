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

# Optionally, convert the CSV file to a database
COPY create_db.py /app/create_db.py  # Copy the script
RUN python /app/create_db.py  # Run the script to create the database

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command to run the app
CMD ["python", "app.py"]
