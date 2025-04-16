from fastapi import HTTPException, Depends
from typing import Any
import asyncio
import datetime
from dotenv import load_dotenv
import ollama

from app.schemas.schema import DebateRequest, DebatePromptResponse, DebateEvaluationResponse
from app.db.singleton import ChromaDBSingleton

load_dotenv()

chroma_client = ChromaDBSingleton()
debate_collection = chroma_client.get_collection()


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


async def generate_debate_prompt(client: Any = Depends(get_model_client)) -> DebatePromptResponse:
    try:
        prompt = (
            "Generate a culturally sensitive real-world ethical dilemma that sparks debate. "
            "The topic should encourage players to take sides and argue with historical, ethical, or empathetic reasoning."
        )
        messages = [{"role": "user", "content": prompt}]
        response = await asyncio.to_thread(client.chat, model="llama3.2:latest", messages=messages)
        content = response["message"]["content"].strip()
        return DebatePromptResponse(prompt=content, timestamp=str(datetime.datetime.now()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dilemma: {str(e)}")


async def evaluate_debate_response(request: DebateRequest, client: Any = Depends(get_model_client)) -> DebateEvaluationResponse:
    try:
        response_embedding = await get_embeddings(request.response, client)

        rag_results = debate_collection.query(
            query_embeddings=[response_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )

        rag_context = ""
        if rag_results and rag_results["documents"] and rag_results["documents"][0]:
            rag_context = "Here are reference arguments for context:\n"
            for i, doc in enumerate(rag_results["documents"][0]):
                rag_context += f"Example {i+1}:\n{doc[:400]}...\n\n"

        prompt = (
            f"Debate Prompt:\n{request.prompt}\n\n"
            f"User's Response:\n{request.response}\n\n"
            f"{rag_context}"
            f"Evaluate the user's response based on:\n"
            f"1. Historical accuracy\n"
            f"2. Ethical reasoning\n"
            f"3. Cultural empathy\n"
            f"Provide a short explanation and a score out of 10 for each category."
        )

        messages = [{"role": "user", "content": prompt}]
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=messages,
            options={"temperature": 0.7}
        )

        return DebateEvaluationResponse(
            evaluation=response["message"]["content"].strip(),
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate debate response: {str(e)}")
