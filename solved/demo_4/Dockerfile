# 1. Base Image: Start from an official, slim Python distribution
FROM python:3.11-slim

# 2. Set Working Directory: All commands below will run inside this folder
WORKDIR /app

RUN pip install --upgrade pip

# 3. Copy Requirements: Copy the requirements file and install dependencies
# We assume you have run 'pip freeze > requirements.txt' previously
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Copy Application Code: Copy our application files
COPY . .

# 5. Expose Port: Tell the outside world the service runs on port 8000
EXPOSE 8000

# 6. Startup Command: Define the default command to run when the container starts
# The host must be 0.0.0.0 to listen to external requests
# There is another approach to this as well.
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python3", "asgi.py"]