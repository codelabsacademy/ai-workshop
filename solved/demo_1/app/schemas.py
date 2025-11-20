from pydantic import BaseModel

class AskGeminiRequest(BaseModel):
    prompt: str
    