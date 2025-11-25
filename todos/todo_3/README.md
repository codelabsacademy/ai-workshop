# 📦 Session 3: Containerization and Orchestration

This hands-on session focuses on turning our Python microservice into a scalable, portable Docker container. We will use a `Dockerfile` to create the image and **Docker Compose** to run the service with persistent configuration.

## 🎯 Learning Goals
- Understand the **Image vs. Container** concepts.

- Write a `Dockerfile` to package our application and dependencies.

- Use `docker build` and `docker run` commands.

- Implement `docker-compose.yml` to manage the service and environment variables in one command.

## 🛠️ Prerequisites & Setup

1. **Code State:**  Ensure you have the final, functional main.py file from Session 1/Extension. Or you can you use the demo-3 in the solved directory.

2. **Software:** **Docker Desktop** must be installed and running on your machine.

3. **API Key:** Your GOOGLE_API_KEY must be accessible in your current shell session.

## Core Concepts: Dockerizing the Service

### 1. The Dockerfile (The Recipe)

```python
# 1. Base Image: Start from an official, slim Python distribution
FROM python:3.11-slim

# 2. Set Working Directory: All commands below will run inside this folder
WORKDIR /app

# 3. Copy Requirements: Copy the requirements file and install dependencies
# We assume you have run 'pip freeze > requirements.txt' previously
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code: Copy our application files
COPY . .

# 5. Expose Port: Tell the outside world the service runs on port 8000
EXPOSE 8000

# 6. Startup Command: Define the default command to run when the container starts
# The host must be 0.0.0.0 to listen to external requests
# There is another approach to this as well.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. The .dockerignore (The Filter) - Good Practice
Create this file to prevent large or sensitive files (like your virtual environment) from being copied into the image context.

```plaintext
venv/
__pycache__/
.git/
.gitignore
```

### 3. Build the Image
This command reads the Dockerfile and creates the static Image (the blueprint).
```bash
docker build -t genai-microservice:v1 .
```

## Orchestration with Docker Compose
We use **Docker Compose** to manage configurations, networking, and environment variables, simplifying the process of running multi-component applications.

### 1. Create `docker-compose.yml`
This YAML file defines how Docker should run our service. It references the image we just built and securely injects the `GOOGLE_API_KEY`.

Using the image you have built:
```yaml
version: '3.8'

services:
  ai-backend:
    # 1. Use the image we built locally
    image: demo_4:v1
    
    # 2. Map host port 8000 to container port 8000
    ports:
      - "8000:8000"
      
    # 3. Environment variables (Critical for API Key)
    # This securely pulls the GOOGLE_API_KEY from your shell environment
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    
    # 4. Set the restart policy
    restart: always
```

#### Alternative docker-compose.yml (Using Build Path) 🏗️
Instead of pre-building the Docker image and referencing it by its tag (demo_4:v1), you can instruct Docker Compose to build the image directly from the local directory every time you run the service.

The key change is replacing the `image:` key with the `build:` key, pointing to the current directory `(.)` where your `Dockerfile` is located.

```yaml
version: '3.8'

services:
  ai-backend:
    # --- CHANGE START ---
    # 1. Instructs Docker Compose to build the image from the Dockerfile 
    #    located in the current directory (.).
    build: 
      context: .
      dockerfile: Dockerfile  # Optional, but good practice to be explicit
    # --- CHANGE END ---
    
    # Optional: You can still tag the image if you want to reference it later
    # image: demo_4:v1 
    
    # 2. Map host port 8000 to container port 8000
    ports:
      - "8000:8000"
      
    # 3. Environment variables (Critical for API Key)
    # This securely pulls the GOOGLE_API_KEY from your shell environment
    environment:
      - GOOGLE_API_KEY=GOOGLE_API_KEY
    
    # 4. Set the restart policy
    restart: always
```

### Why Use the build Path?
**Simplified Workflow:** You only need one command, docker compose up -d, which handles both the image building and container running. This is ideal for local development.

**Automatic Updates:** Docker Compose automatically detects changes in your Dockerfile or any files copied into the image, ensuring you are always running the latest code.

**Reduced Errors:** You avoid the possibility of forgetting to run the docker build -t demo_4:v1 . command before trying to run the container.

### 2. Run and Manage with Compose

These commands simplify the entire lifecycle management:

```bash
# Start the service using Docker Compose (runs in detached mode)
docker compose up -d

# Check the logs to verify startup (optional)
docker compose logs -f ai-backend

# Stop and remove the container
docker compose down
```

## Dockerizing Demo: Multi-Tool Agent Service (Session 3 Hands-On)
This section contains the precise instructions for participants to execute the containerization process on the solved **demo 3** application.

| Step | Action | Command/Instruction |
| :--- | :--- | :--- |
| **1. Preparation** | Ensure you have a `requirements.txt` file created from your current virtual environment. | `pip freeze > requirements.txt` |
| **2. Build Image** | Build the container image named `demo_4:v1`. | `docker build -t demo_4:v1 .` |
| **3. Run Service** | Start the service using the Docker Compose configuration. | `docker compose up -d` |
| **4. Verification** | Open the browser and verify the service is running. | Open `http://localhost:8000/docs` and execute the `/ask-ai` endpoint. |
| **5. Clean Up** | Stop and remove the running container. | `docker compose down` |