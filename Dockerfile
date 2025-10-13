# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libgl1
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application's code into the container at /app
COPY main.py .
COPY config.py .
COPY database.py .
COPY security.py .
COPY retraining_worker.py .
COPY train.py .
COPY evaluate.py .
COPY run_pipeline.py .
COPY dataset.yaml .
COPY routers ./routers
COPY models ./models

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]