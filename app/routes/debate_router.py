from typing import Any

from fastapi import APIRouter, Depends

from app.controllers.debate_controller import (
    generate_debate_prompt,
    evaluate_debate_response,
    process_debate_message, get_model_client
)
from app.schemas.schema import (
    DebateRequest,
    DebatePromptResponse,
    DebateEvaluationResponse,
    DebateMessageRequest,
    DebateMessageResponse
)

debate_router = APIRouter(prefix="/debate", tags=["debate"])

@debate_router.get("/prompt", response_model=DebatePromptResponse)
async def get_debate_prompt(client : Any = Depends(get_model_client)):
    """Get a new ethical dilemma for debate"""
    return await generate_debate_prompt(client)

@debate_router.post("/message", response_model=DebateMessageResponse)
async def send_debate_message(request: DebateMessageRequest, client : Any = Depends(get_model_client)):
    """Send a message in the debate conversation and get AI response"""
    return await process_debate_message(request,client)

@debate_router.post("/evaluate", response_model=DebateEvaluationResponse)
async def evaluate_debate(request: DebateRequest, client : Any = Depends(get_model_client)):
    """Evaluate the entire debate conversation"""
    return await evaluate_debate_response(request,client)