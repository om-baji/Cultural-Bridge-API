from typing import Any

from app.controllers.role_playing import generate_role_play, get_model_client
from app.schemas.schema import StoryResponse, RolePlayRequest
from fastapi import APIRouter, Depends

rpg_router = APIRouter()

@rpg_router.post("/rpg_mode", response_model=StoryResponse)
async def rpg_endpoint(request : RolePlayRequest, client : Any = Depends(get_model_client)):
    return await generate_role_play(request, client)
