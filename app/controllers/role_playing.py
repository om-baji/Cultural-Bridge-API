from fastapi import HTTPException
from typing import Any, List, Dict
import ollama
import asyncio
from dotenv import load_dotenv
import datetime

from app.schemas.schema import RolePlayRequest, StoryResponse
from app.db.singleton import ChromaDBSingleton

load_dotenv()

chroma_client = ChromaDBSingleton()
chroma_collection = chroma_client.get_collection()


async def get_model_client():
    return ollama


async def get_embeddings(text: str, client: Any = None):
    if client is None:
        client = await get_model_client()

    response = await asyncio.to_thread(
        client.embeddings,
        model="all-minilm:33m",
        prompt=text
    )
    return response["embedding"]


async def generate_role_play(request: RolePlayRequest, client: Any):
    try:
        # Set up system prompt to guide AI behavior
        system_prompt = (
            f"You are role-playing as a {request.role} from the {request.culture} culture, "
            f"during the {request.era} era. You respond based on that role only. "
            f"Maintain historical and cultural accuracy. Use a {request.tone} tone and write in {request.language}. "
            f"{'Include emotional and reflective thoughts as well.' if request.include_emotion else ''}"
        )

        # Convert past interactions into chat history
        history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in request.chat_history:
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": turn["ai"]})

        # Add current user input
        history.append({"role": "user", "content": request.user_input})

        # Call Ollama with the message history
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=history,
            options={"temperature": 0.75, "top_p": 0.9}
        )

        reply = response['message']['content'].strip()
        embeddings = await get_embeddings(reply, client)

        metadata = {
            "mode": "role-play",
            "culture": request.culture,
            "role": request.role,
            "era": request.era,
            "tone": request.tone,
            "language": request.language
        }

        chroma_collection.add(
            documents=[reply],
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[request.role + "-role-" + str(datetime.datetime.now().timestamp())]
        )

        return StoryResponse(
            story=reply,
            character_count=len(reply),
            language=request.language,
            metadata=metadata,
            used_rag=False,
            reference_count=0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in role-play generation: {str(e)}")
