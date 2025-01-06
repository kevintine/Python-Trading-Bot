# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container (ensure all dependencies are listed here)
COPY requirements.txt /app/
COPY TA_Lib-0.4.24-cp310-cp310-win_amd64.whl /app/

# Install the dependencies
RUN pip install --upgrade pip \
    && pip install /app/TA_Lib-0.4.24-cp310-cp310-win_amd64.whl \
    && pip install -r requirements.txt

# Copy the rest of your project files into the container
COPY . /app/

# Set the default directory to where app.py is located
WORKDIR /app/web-app.py

# Expose the port your app runs on (change if necessary)
EXPOSE 5000

# Command to run your application
CMD ["python", "app.py"]
