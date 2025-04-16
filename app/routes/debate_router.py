from fastapi import APIRouter, Depends
from app.controllers.debate_controller import generate_debate_prompt, evaluate_debate_response, get_model_client
from app.schemas.schema import DebatePromptResponse, DebateRequest, DebateEvaluationResponse

debaterouter = APIRouter()

@debaterouter.get("/debate/prompt", response_model=DebatePromptResponse)
async def generate_prompt_endpoint(client=Depends(get_model_client)):
    return await generate_debate_prompt(client)

@debaterouter.post("/debate/evaluate", response_model=DebateEvaluationResponse)
async def evaluate_response_endpoint(request: DebateRequest, client=Depends(get_model_client)):
    return await evaluate_debate_response(request, client)
