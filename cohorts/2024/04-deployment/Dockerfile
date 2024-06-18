# Use the base image with Python 3.10.13 slim version
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script and model files into the container
COPY test.py .
COPY model.bin .

# Install specific versions of dependencies
RUN pip install scikit-learn==1.0.2 numpy==1.21.4 pandas pyarrow fastparquet

# Command to run the Python script
CMD ["python", "test.py", "--year", "2023", "--month", "5"]

