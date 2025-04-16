from typing import Any

from app.controllers.conflict_resolution import generate_conflict_scenario, get_model_client
from app.schemas.schema import ConflictResponse, ConflictRequest
from fastapi import APIRouter, Depends

conflict_router = APIRouter()

@conflict_router.post("/conflict_resolution", response_model=ConflictResponse)
async def conflict_resolution_endpoint(request: ConflictRequest, client: Any = Depends(get_model_client)):
    return await generate_conflict_scenario(request, client)