from fastapi import APIRouter, Depends
from app.controllers.story import get_model_client, add_story, generate_story
from app.schemas.schema import StoryResponse, StoryRequest, Response

router = APIRouter()

@router.post("/add_story", response_model=Response)
async def add_story_endpoint(request: StoryRequest):
    return await add_story(request)

@router.post("/story", response_model=StoryResponse)
async def generate_story_endpoint(request: StoryRequest, client=Depends(get_model_client)):
    return await generate_story(request, client)
