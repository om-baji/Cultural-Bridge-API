from fastapi import HTTPException
from typing import Any, List, Dict
import ollama
import asyncio
from dotenv import load_dotenv
import datetime
import uuid
import re

from app.schemas.schema import ConflictRequest, ConflictResponse, KalkiScore
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


async def generate_conflict_scenario(request: ConflictRequest, client: Any):
    try:
        conflict_context = {
            "india_pakistan": "the 1947 India-Pakistan partition with tension over borders, refugees, and religious differences",
            "israeli_palestinian": "the Israeli-Palestinian conflict with disputes over territory, security, and self-determination",
            "indigenous_rights": "Indigenous rights movements facing challenges of land rights, sovereignty, and cultural preservation",
            "northern_ireland": "the Northern Ireland conflict (The Troubles) with tension between unionists and nationalists",
            "rwanda": "the ethnic tensions in Rwanda leading up to and following the 1994 genocide"
        }

        faction_description = {
            "side_a": "representing the first main party in the conflict",
            "side_b": "representing the second main party in the conflict",
            "neutral": "as a neutral third party attempting to facilitate peace"
        }

        # Set up system prompt to guide AI behavior
        system_prompt = (
            f"You are simulating a conflict resolution scenario for {conflict_context.get(request.conflict_type, 'a historical conflict')}. "
            f"The user is playing as a {request.player_role} {faction_description.get(request.player_faction, '')}. "
            f"Current tension level is {request.tension_level}/100. "
            f"Provide realistic consequences to the user's actions, detailing how they affect the conflict. "
            f"Include decisions other parties might make in response. "
            f"If the user makes choices that would realistically escalate tensions, reflect that in your response. "
            f"If they make de-escalatory choices, show progress toward resolution. "
            f"Maintain historical accuracy while allowing for counterfactual scenarios based on user choices."
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
            model="llama3:latest",
            messages=history,
            options={"temperature": 0.7, "top_p": 0.9}
        )

        reply = response['message']['content'].strip()

        # Determine new tension level based on AI response
        new_tension = calculate_tension(request.tension_level, reply, request.user_input)

        # Determine if scenario has reached a conclusion
        is_concluded = check_conclusion(new_tension, request.current_stage)

        # Generate next available actions
        next_actions = generate_next_actions(new_tension, request.player_faction, request.player_role)

        # Calculate KALKI score if concluded
        kalki_score = None
        if is_concluded:
            kalki_score = await calculate_kalki_score(request.chat_history, request.user_input, reply, client)

        # Store interaction in vector database
        embeddings = await get_embeddings(reply, client)
        metadata = {
            "mode": "conflict-resolution",
            "conflict_type": request.conflict_type,
            "role": request.player_role,
            "faction": request.player_faction,
            "tension_level": new_tension,
            "stage": request.current_stage
        }

        chroma_collection.add(
            documents=[reply],
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[f"conflict-{request.session_id}-{str(datetime.datetime.now().timestamp())}"]
        )

        return ConflictResponse(
            response=reply,
            tension_level=new_tension,
            current_stage=request.current_stage + (0 if not is_concluded else 1),
            available_actions=next_actions,
            is_concluded=is_concluded,
            metadata=metadata,
            session_id=request.session_id or str(uuid.uuid4()),
            kalki_score=kalki_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in conflict simulation: {str(e)}")


async def calculate_kalki_score(chat_history: List[Dict[str, str]], user_input: str, ai_response: str,
                                client: Any) -> KalkiScore:
    # Combine all conversation for context
    full_conversation = ""
    for turn in chat_history:
        full_conversation += f"User: {turn['user']}\nAI: {turn['ai']}\n\n"
    full_conversation += f"User: {user_input}\nAI: {ai_response}"

    # Set up prompt for KALKI scoring evaluation
    evaluation_prompt = (
        "Evaluate the user's conflict resolution approach based on the KALKI scoring system:\n\n"
        "1. EMPATHY (0-30): Did the user consider multiple perspectives? Score higher if they demonstrated understanding of all sides.\n"
        "2. DIPLOMATIC SKILL (0-30): Did the user promote peaceful negotiation? Score higher for constructive dialogue and compromise.\n"
        "3. HISTORICAL ACCURACY (0-20): Were the user's decisions informed by real-world lessons? Score higher for realistic approaches.\n"
        "4. ETHICAL BALANCE (0-20): Did the user avoid bias and maintain ethical principles? Score higher for fair solutions.\n\n"
        "Based on the conversation below, provide numeric scores for each category and a brief explanation of each score.\n\n"
        f"{full_conversation}\n\n"
        "Respond in this exact format (with ONLY the scores and no additional text):\n"
        "EMPATHY: [score]\n"
        "DIPLOMATIC_SKILL: [score]\n"
        "HISTORICAL_ACCURACY: [score]\n"
        "ETHICAL_BALANCE: [score]\n"
    )

    # Get evaluation from LLM
    eval_response = await asyncio.to_thread(
        client.generate,
        model="llama3:latest",
        prompt=evaluation_prompt,
        options={"temperature": 0.2}
    )

    eval_text = eval_response['response'].strip()

    # Extract scores using regex
    empathy_match = re.search(r"EMPATHY: (\d+)", eval_text)
    diplomatic_match = re.search(r"DIPLOMATIC_SKILL: (\d+)", eval_text)
    historical_match = re.search(r"HISTORICAL_ACCURACY: (\d+)", eval_text)
    ethical_match = re.search(r"ETHICAL_BALANCE: (\d+)", eval_text)

    empathy = int(empathy_match.group(1)) if empathy_match else 15
    diplomatic = int(diplomatic_match.group(1)) if diplomatic_match else 15
    historical = int(historical_match.group(1)) if historical_match else 10
    ethical = int(ethical_match.group(1)) if ethical_match else 10

    # Ensure scores are within range
    empathy = max(0, min(30, empathy))
    diplomatic = max(0, min(30, diplomatic))
    historical = max(0, min(20, historical))
    ethical = max(0, min(20, ethical))

    total = empathy + diplomatic + historical + ethical

    return KalkiScore(
        empathy=empathy,
        diplomatic_skill=diplomatic,
        historical_accuracy=historical,
        ethical_balance=ethical,
        total_score=total
    )


def calculate_tension(current_tension: int, ai_response: str, user_input: str) -> int:
    lower_response = ai_response.lower()
    lower_input = user_input.lower()

    # Keywords that would indicate tension is increasing
    escalation_words = ["violence", "attack", "protest", "conflict", "tension", "hostility",
                        "disagree", "reject", "refuse", "militant", "military", "force"]

    # Keywords that would indicate tension is decreasing
    deescalation_words = ["peace", "agreement", "compromise", "negotiate", "cooperate",
                          "collaborate", "understand", "reconcile", "dialogue", "diplomacy"]

    # Calculate change based on keyword presence
    escalation_count = sum(word in lower_response or word in lower_input for word in escalation_words)
    deescalation_count = sum(word in lower_response or word in lower_input for word in deescalation_words)

    tension_change = (escalation_count - deescalation_count) * 5

    # Apply change and ensure within bounds
    new_tension = max(0, min(100, current_tension + tension_change))
    return new_tension


def check_conclusion(tension: int, stage: int) -> bool:
    # Scenario concludes if either tension reaches extremes or stage is high enough
    if tension <= 10 or tension >= 90:
        return True
    if stage >= 5:  # Arbitrary max stages before conclusion
        return True
    return False


def generate_next_actions(tension: int, faction: str, role: str) -> List[str]:
    common_actions = ["Negotiate", "Make public statement", "Propose solution"]

    if tension > 70:
        if faction in ["side_a", "side_b"]:
            return common_actions + ["Seek international support", "Show of force"]
        else:
            return common_actions + ["Call emergency meeting", "Propose sanctions"]
    elif tension < 30:
        return common_actions + ["Form joint committee", "Celebrate progress"]
    else:
        return common_actions + ["Hold private talks", "Request mediation"]