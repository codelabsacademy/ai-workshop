from fastapi import FastAPI
from routes import ask

app = FastAPI(title="Gemini Microservice")

app.include_router(ask.router)  