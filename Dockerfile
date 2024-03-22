# Use the official Python image as a base
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install -r requirements.txt

# Copy the Flask application code into the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the default command to run when the container starts
CMD ["python", "app.py"]
