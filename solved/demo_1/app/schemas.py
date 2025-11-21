from pydantic import BaseModel, Field
from typing import List

class AskGeminiRequest(BaseModel):
    prompt: str

class InvestmentProfile(BaseModel):
    """
    A schema for structured JSON output of key investment data.
    """
    carbon_intensity: float = Field(description="The calculated carbon intensity (CO2e/revenue) score of the company.")
    board_diversity_score: float = Field(description="The mandatory board diversity score (0.0 to 1.0).")
    key_risks: List[str] = Field(description="A list of three critical sustainability or compliance risks identified.")
    compliance_status: str = Field(description="Overall compliance status: 'COMPLIANT' or 'FLAGGED'.")