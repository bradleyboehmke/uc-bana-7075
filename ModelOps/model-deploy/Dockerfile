# Use official Python image
FROM docker-all.artifactory.8451.cloud/core-images/python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy all application files to the container
COPY . /app

# Security stuff
USER root

# Install dependencies
RUN python3.12 -m pip install --upgrade pip setuptools
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Security stuff
USER e451

# Expose necessary ports for FastAPI (5000) and Streamlit (8501)
EXPOSE 5000 8501

# Start both FastAPI and Streamlit using the Python script
CMD ["python3.12", "fastapi_streamlit.py"]
