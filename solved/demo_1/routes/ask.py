from fastapi import APIRouter
from app.schemas import AskGeminiRequest

from controllers.gemini import Gemini


router = APIRouter(prefix="/v1/ask-ai", tags=["Ask Gemini"])

@router.post("")
def ask_ai(request: AskGeminiRequest):
    """
    Sends the user's prompt to Gemini via LangChain and returns the text response.
    """

    data = Gemini().ask_gemini(request.prompt)

    return data

