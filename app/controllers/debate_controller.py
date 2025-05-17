import logging

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


async def evaluate_debate_response(request: DebateRequest, client: Any) -> DebateEvaluationResponse:
    try:
        response_embedding = await get_embeddings(request.user_response, client)

        rag_results = await asyncio.to_thread(
            debate_collection.query,
            query_embeddings=[response_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )

        rag_context = ""
        if rag_results.get("documents") and rag_results["documents"][0]:
            rag_context = "Here are reference arguments for context:\n"
            for i, doc in enumerate(rag_results["documents"][0]):
                rag_context += f"Example {i + 1}:\n{doc[:400]}...\n\n"

        prompt = (
            f"Debate Prompt:\n{request.prompt}\n\n"
            f"Debate Response:\n{request.user_response}\n\n"
            f"{rag_context}"
            f"Evaluate the debate based on:\n"
            f"1. Historical accuracy\n"
            f"2. Ethical reasoning\n"
            f"3. Cultural empathy\n"
            f"4. Logical structure\n"
            f"5. Evidence-based reasoning\n\n"
            f"Give a short evaluation.\n"
            f"Then provide a score from 0-10 for each criteria.\n"
            f"Finally, suggest one improvement idea.\n\n"
            f"Format your answer exactly as:\n"
            f"Evaluation: <text>\n"
            f"Scores:\n"
            f"Historical accuracy: <number>\n"
            f"Ethical reasoning: <number>\n"
            f"Cultural empathy: <number>\n"
            f"Logical structure: <number>\n"
            f"Evidence-based reasoning: <number>\n"
            f"Suggestion: <text>"
        )

        messages = [{"role": "user", "content": prompt}]

        response = await asyncio.to_thread(
            client.chat,
            model="llama3.2:latest",
            messages=messages,
            options={"temperature": 0.4}
        )

        content = response["message"]["content"]

        lines = content.splitlines()
        eval_text = ""
        scores = {}
        suggestion = None

        mode = None
        for line in lines:
            if line.lower().startswith("evaluation:"):
                mode = "evaluation"
                eval_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("scores:"):
                mode = "scores"
            elif mode == "scores" and ":" in line:
                key, val = line.split(":", 1)
                scores[key.strip().lower()] = float(val.strip())
            elif line.lower().startswith("suggestion:"):
                suggestion = line.split(":", 1)[1].strip()

        return DebateEvaluationResponse(
            evaluation=eval_text,
            scores=scores,
            suggestions=suggestion,
            timestamp=str(datetime.datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate debate response: {str(e)}")
