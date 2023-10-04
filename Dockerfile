# Use a base image with Python and libraries pre-installed
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install any Python dependencies (specify your requirements.txt)
RUN pip install -r requirements.txt

# Expose a port if needed (e.g., for API service)
# EXPOSE 8080

# Define the command to run your Python script
CMD ["python", "train_model.py"]
