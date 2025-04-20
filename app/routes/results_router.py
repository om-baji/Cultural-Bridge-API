from fastapi import APIRouter, Depends, HTTPException
from typing import Any

from app.controllers.results import analyze_user_responses
from app.controllers.role_playing import get_model_client
from app.schemas.schema import ResultsResponse

results_router = APIRouter()


@results_router.get("/results/{user_id}", response_model=ResultsResponse)
async def get_user_results(user_id: str, client: Any = Depends(get_model_client)):
    """
    Get a comprehensive analysis of the user's roleplay performance using the KALKI scoring system.

    - Empathy (0-30): Did the user consider multiple perspectives?
    - Diplomatic Skill (0-30): Did the responses promote peaceful negotiation?
    - Historical Accuracy (0-20): Were the responses based on real-world historical lessons?
    - Ethical Balance (0-20): Did the user avoid bias and consider ethical implications?

    Returns detailed scores, feedback, and improvement suggestions.
    """
    return await analyze_user_responses(user_id, client)