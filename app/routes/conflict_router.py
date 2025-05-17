from fastapi import APIRouter, Depends, HTTPException
from typing import Any

from app.controllers.conflict_resolution import get_model_client, generate_conflict_scenario
from app.schemas.schema import ConflictRequest, ConflictResponse, Response

conflict_router = APIRouter()


@conflict_router.post("/start-conflict", response_model=ConflictResponse)
async def start_conflict_endpoint(request: ConflictRequest, client: Any = Depends(get_model_client)):
    """
    Endpoint to start a new conflict resolution scenario.
    This should be used for the initial request when beginning a new scenario.
    """
    if request.chat_history is None:
        request.chat_history = []

    request.session_id = request.session_id or None
    request.current_stage = request.current_stage or 0

    try:
        return await generate_conflict_scenario(request, client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting conflict scenario: {str(e)}")


@conflict_router.post("/continue-conflict", response_model=ConflictResponse)
async def continue_conflict_endpoint(request: ConflictRequest, client: Any = Depends(get_model_client)):
    """
    Endpoint to continue an existing conflict resolution scenario.
    This should be used for subsequent requests after starting a scenario.
    """
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required to continue a conflict scenario")

    if request.chat_history is None:
        request.chat_history = []

    try:
        return await generate_conflict_scenario(request, client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error continuing conflict scenario: {str(e)}")
# from fastapi import APIRouter, Depends, HTTPException
# from typing import Any, Optional
# import redis.asyncio as redis
# import json
# import os
# from uuid import uuid4
#
# from app.schemas.schema import ConflictRequest, ConflictResponse
# from app.controllers.conflict_resolution import generate_conflict_scenario, get_model_client
#
# redis_client = redis.Redis(
#     host=os.getenv("REDIS_HOST", "localhost"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD", ""),
#     decode_responses=True
# )
#
# conflict_router = APIRouter()
#
#
# @conflict_router.post("/conflict_resolution", response_model=ConflictResponse)
# async def conflict_resolution_endpoint(request_data: dict, client: Any = Depends(get_model_client)):
#     try:
#         if request_data.get("action") == "start":
#             scenario_id = request_data.get("scenario_id")
#             role_id = request_data.get("role_id")
#
#             conflict_mapping = {
#                 "india_pakistan": "india_pakistan",
#                 "israeli_palestinian": "israeli_palestinian",
#                 "indigenous_rights": "indigenous_rights",
#                 "northern_ireland": "northern_ireland",
#                 "rwanda": "rwanda"
#             }
#
#             role_mapping = {
#                 "side_a": {"faction": "side_a", "role": "diplomat"},
#                 "side_b": {"faction": "side_b", "role": "diplomat"},
#                 "neutral": {"faction": "neutral", "role": "mediator"}
#             }
#
#             conflict_type = conflict_mapping.get(scenario_id)
#             role_data = role_mapping.get(role_id, {"faction": "neutral", "role": "diplomat"})
#
#             session_id = str(uuid4())
#
#             conflict_request = ConflictRequest(
#                 conflict_type=conflict_type,
#                 player_faction=role_data["faction"],
#                 player_role=role_data["role"],
#                 tension_level=50,
#                 current_stage=0,
#                 chat_history=[],
#                 user_input="Start scenario",
#                 session_id=session_id
#             )
#
#             response = await generate_conflict_scenario(conflict_request, client)
#
#             session_data = {
#                 "conflict_type": conflict_type,
#                 "player_faction": role_data["faction"],
#                 "player_role": role_data["role"],
#                 "tension_level": response.tension_level,
#                 "current_stage": response.current_stage,
#                 "chat_history": [],
#                 "session_id": session_id
#             }
#
#             await redis_client.set(f"conflict_session:{session_id}", json.dumps(session_data), ex=3600)
#
#             return response
#
#         elif request_data.get("action") == "respond":
#             user_response = request_data.get("response", "")
#             session_id = request_data.get("session_id")
#
#             if not session_id:
#                 raise HTTPException(status_code=400, detail="Missing session_id")
#
#             session_data_str = redis_client.get(f"conflict_session:{session_id}")
#             if not session_data_str:
#                 raise HTTPException(status_code=404, detail="Session not found or expired")
#
#             session_data = json.loads(session_data_str)
#
#             conflict_request = ConflictRequest(
#                 conflict_type=session_data["conflict_type"],
#                 player_faction=session_data["player_faction"],
#                 player_role=session_data["player_role"],
#                 tension_level=session_data["tension_level"],
#                 current_stage=session_data["current_stage"],
#                 chat_history=session_data["chat_history"],
#                 user_input=user_response,
#                 session_id=session_id
#             )
#
#             response = await generate_conflict_scenario(conflict_request, client)
#
#             updated_chat_history = session_data["chat_history"] + [{"user": user_response, "ai": response.response}]
#
#             updated_session_data = {
#                 "conflict_type": session_data["conflict_type"],
#                 "player_faction": session_data["player_faction"],
#                 "player_role": session_data["player_role"],
#                 "tension_level": response.tension_level,
#                 "current_stage": response.current_stage,
#                 "chat_history": updated_chat_history,
#                 "session_id": session_id
#             }
#
#             await redis_client.set(f"conflict_session:{session_id}", json.dumps(updated_session_data), ex=3600)
#
#             return response
#         else:
#             raise HTTPException(status_code=400, detail="Invalid action specified")
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
#
#
# @conflict_router.delete("/conflict_session/{session_id}")
# async def delete_conflict_session(session_id: str):
#     if redis_client.exists(f"conflict_session:{session_id}"):
#         await redis_client.delete(f"conflict_session:{session_id}")
#         return {"status": "success", "message": "Session deleted"}
#     return {"status": "error", "message": "Session not found"}