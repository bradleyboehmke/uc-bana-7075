# Use official Python image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy all application files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose necessary ports for FastAPI (5000) and Streamlit (8501)
EXPOSE 5000 8501

# Start both FastAPI and Streamlit using the Python script
CMD ["python", "fastapi_streamlit.py"]
