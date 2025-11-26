from app.main import app
import uvicorn
import os

if __name__ == "__main__":
    # starting the ASGI server
    uvicorn.run(app, host="0.0.0.0", port=8000)