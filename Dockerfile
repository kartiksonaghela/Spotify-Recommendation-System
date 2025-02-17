# Setup base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py /app/
COPY pipeline/ /app/pipeline/
COPY data /app/data  

# Expose port for Streamlit
EXPOSE 8000

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
