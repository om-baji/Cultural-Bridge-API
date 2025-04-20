from fastapi import HTTPException, Depends
from typing import Any, List, Dict
import asyncio
import datetime
from dotenv import load_dotenv
import ollama

from app.schemas.schema import (
    DebatePromptResponse,
    DebateEvaluationResponse,
    DebateRequest,
    DebateMessageRequest,
    DebateMessageResponse
)
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


async def generate_debate_prompt(client: Any) -> DebatePromptResponse:
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


async def process_debate_message(
        request: DebateMessageRequest,
        client: Any
) -> DebateMessageResponse:
    """
    Process a new message in the debate conversation and return the AI's response
    """
    try:
        # Format the context for the AI based on conversation history
        messages = [{"role": "system", "content": (
            f"You are an AI debate partner discussing the following ethical dilemma:\n{request.prompt}\n\n"
            f"Maintain a thoughtful, challenging stance in the debate. "
            f"Consider ethical principles, cultural contexts, and historical precedents in your reasoning."
        )}]

        # Add conversation history
        messages.extend(request.history)

        # Add the user's new message
        messages.append({"role": "user", "content": request.message})

        # Generate AI response
        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=messages,
            options={"temperature": 0.8}
        )

        ai_response = response["message"]["content"].strip()

        # Save this conversation to the database for future reference
        # This could be implemented later to build a knowledge base

        return DebateMessageResponse(
            content=ai_response,
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process debate message: {str(e)}")


async def evaluate_debate_response(request: DebateRequest,
                                   client: Any ) -> DebateEvaluationResponse:
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
                rag_context += f"Example {i + 1}:\n{doc[:400]}...\n\n"

        prompt = (
            f"Debate Prompt:\n{request.prompt}\n\n"
            f"Full Debate Conversation:\n{request.response}\n\n"
            f"{rag_context}"
            f"Evaluate the quality of arguments in this debate based on:\n"
            f"1. Historical accuracy\n"
            f"2. Ethical reasoning\n"
            f"3. Cultural empathy\n"
            f"4. Logical structure\n"
            f"5. Evidence-based reasoning\n\n"
            f"For each participant in the debate, provide strengths and areas for improvement. "
            f"Give specific examples from the debate to support your evaluation. "
            f"Conclude with overall feedback and suggestions for more effective debate techniques."
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