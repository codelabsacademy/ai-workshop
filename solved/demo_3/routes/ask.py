from fastapi import APIRouter
from app.schemas import AskGeminiRequest

from controllers.gemini import Gemini
from controllers.agent import AgentController


router = APIRouter(prefix="/v1/ask-ai", tags=["Ask Gemini"])

@router.post("")
def ask_ai(request: AskGeminiRequest):
    """
    Sends the user's prompt to Gemini via LangChain and returns the text response.
    """

    data = Gemini().ask_gemini(request.prompt)

    return data

@router.post("/extract-profile")
async def extract_policy(request: AskGeminiRequest):
    
    data = await Gemini(temperature=0).extract_profile(request.prompt)

    return data

@router.post("agent")
async def ask_agent(request: AskGeminiRequest):
    
    data = await AgentController().ask_agent(request.prompt)

    return data