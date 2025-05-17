from typing import Any

from app.controllers.role_playing import generate_role_play, get_model_client, evaluate_chat_history
from app.schemas.schema import StoryResponse, RolePlayRequest, EvaluationResponse, EvaluationRequest
from fastapi import APIRouter, Depends

rpg_router = APIRouter()

@rpg_router.post("/rpg_mode", response_model=StoryResponse)
async def rpg_endpoint(request : RolePlayRequest, client : Any = Depends(get_model_client)):
    return await generate_role_play(request, client)

@rpg_router.post("/rpg_evaluate", response_model=EvaluationResponse)
async def evaluation_endpoint(request: EvaluationRequest, client: Any = Depends(get_model_client)):
    return await evaluate_chat_history(request, client)